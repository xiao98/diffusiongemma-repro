"""PyTorch port of DiffusionGemma's stage-1 SFT and block sampler onto a small HF causal LM.

Every piece names the official file it mirrors (third_party/gemma/gemma/diffusion/...):
  SelfConditioning        <- _transformer.py: SelfConditioning
  encode_logits           <- gm/nn/gemma4/_modules.py: Embedder.encode_logits
  sft_loss                <- hackable_diffusion_adapter/hd/sft_model.py: SFTDiffusion.__call__
  _decoder_mask           <- hackable_diffusion_adapter/hd/mask_helpers.py: create_decoder_attention_mask
  sample_from_predictions <- _sampler.py: SampleFromPredictions
  generate                <- _sampler.py: DiffusionSampler + _early_stopping.py + _chat_sampler.py defaults

Deviations from the official code are listed in README.md.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Gemma RMSNorm: x / rms(x) * (1 + scale), scale initialised at zero."""

    def __init__(self, dim, with_scale=True):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(dim)) if with_scale else None

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)
        return x if self.scale is None else x * (1 + self.scale)


class SelfConditioning(nn.Module):
    """post_norm(canvas_emb + FFW(pre_norm(sc_signal))), FFW = Gemma's gated GELU block.

    post_norm sets the RMS to 1, which matches Gemma's sqrt(d)-scaled embeddings. Other
    base models have a different embedding scale (SmolLM2: RMS 0.13), so the output is
    rescaled to out_scale = the base model's own embedding RMS. Without this the decoder
    input is ~7.6x larger than anything the pretrained weights have seen.
    """

    def __init__(self, features, hidden_dim, out_scale):
        super().__init__()
        self.register_buffer("out_scale", torch.tensor(float(out_scale)))
        self.pre_norm = RMSNorm(features)
        self.gating = nn.Linear(features, 2 * hidden_dim, bias=False)
        self.linear = nn.Linear(hidden_dim, features, bias=False)
        self.post_norm = RMSNorm(features, with_scale=False)

    def forward(self, canvas_emb, sc_signal):
        a, b = self.gating(self.pre_norm(sc_signal)).chunk(2, dim=-1)
        return self.post_norm(canvas_emb + self.linear(F.gelu(a) * b)) * self.out_scale


def _entropy(logits):
    logp = F.log_softmax(logits.float(), -1)
    return -(logp.exp() * logp).sum(-1)


def sample_from_predictions(logits, entropy_bound, final_step, vocab_size):
    """Accept the lowest-entropy tokens while the entropy accumulated *before* each one
    stays <= entropy_bound; renoise every other position with a fresh uniform token."""
    tokens = torch.distributions.Categorical(logits=logits.float()).sample()
    sorted_ent, order = _entropy(logits).sort(-1)
    accept_sorted = (sorted_ent.cumsum(-1) - sorted_ent) <= entropy_bound
    accept = torch.zeros_like(accept_sorted).scatter(-1, order, accept_sorted)
    if final_step:
        accept = torch.ones_like(accept)
    return torch.where(accept, tokens, torch.randint_like(tokens, vocab_size)), accept


class DiffusionLM(nn.Module):
    """One set of weights used two ways: causal encoder (context) and bidirectional
    denoising decoder (current canvas)."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        emb = lm.get_input_embeddings()
        self.vocab_size, dim = emb.weight.shape
        # Gemma scales embeddings by sqrt(d) inside its embedding layer; others don't.
        self.embed_scale = float(getattr(emb, "embed_scale", 1.0))
        hidden = getattr(lm.config, "intermediate_size", 4 * dim)
        emb_rms = emb.weight.detach().float().pow(2).mean(-1).sqrt().mean() * self.embed_scale
        self.self_conditioner = SelfConditioning(dim, hidden, emb_rms)

    def embed(self, tokens):
        return self.lm.get_input_embeddings()(tokens)

    def encode_logits(self, logits):
        w = self.lm.get_input_embeddings().weight
        return (F.softmax(logits.float(), -1).to(w.dtype) @ w) * self.embed_scale

    def head(self, hidden):
        return self.lm.get_output_embeddings()(hidden)

    # ---------------------------------------------------------------- training

    @staticmethod
    def _decoder_mask(sel, prompt_len, canvas_size, total_len):
        """[B,1,L,L] additive mask. Context rows are causal. Rows of the selected canvas
        see the prompt plus every canvas <= selected (so: bidirectional inside it)."""
        pos = torch.arange(total_len, device=sel.device)
        start = prompt_len + sel * canvas_size  # [B]
        end = start + canvas_size
        causal = pos[None, :] <= pos[:, None]  # [L,L]
        in_sel = (pos[None, :] >= start[:, None]) & (pos[None, :] < end[:, None])  # [B,L]
        visible = pos[None, :] < end[:, None]  # [B,L]
        allowed = torch.where(in_sel[:, :, None], visible[:, None, :], causal[None])
        return allowed[:, None], in_sel

    def _decode_selected(self, prompt, x0, xt, sel, sc_logits, canvas_size):
        """Denoiser logits [B, canvas_size, V] for each example's selected canvas."""
        B, P = prompt.shape
        L = P + x0.shape[1]
        allowed, in_sel = self._decoder_mask(sel, P, canvas_size, L)
        clean = self.embed(torch.cat([prompt, x0], 1))
        noisy = self.embed(torch.cat([prompt, xt], 1))
        idx = (P + sel * canvas_size)[:, None] + torch.arange(canvas_size, device=sel.device)
        gather = idx[:, :, None].expand(-1, -1, clean.shape[-1])
        canvas_emb = noisy.gather(1, gather)
        sc = torch.zeros_like(canvas_emb) if sc_logits is None else self.encode_logits(sc_logits)
        emb = clean.scatter(1, gather, self.self_conditioner(canvas_emb, sc).to(clean.dtype))
        mask = torch.zeros(allowed.shape, dtype=emb.dtype, device=emb.device)
        mask.masked_fill_(~allowed, torch.finfo(emb.dtype).min)
        hidden = self.lm.base_model(inputs_embeds=emb, attention_mask=mask).last_hidden_state
        return self.head(hidden.gather(1, gather)), idx

    def sft_loss(self, prompt, x0, canvas_size, self_cond_prob=0.5):
        """prompt [B,P], x0 [B, K*canvas_size] clean response tokens.
        Returns (encoder AR loss, decoder diffusion loss); official weights are 1.0 each."""
        B = x0.shape[0]
        num_canvases = x0.shape[1] // canvas_size
        dev = x0.device

        # Uniform (multinomial) corruption, linear schedule: each token is replaced by a
        # uniformly random token with probability t, t ~ U[1e-4, 1-1e-4] per example.
        t = torch.rand(B, device=dev) * (1 - 2e-4) + 1e-4
        noise = torch.rand(x0.shape, device=dev) < t[:, None]
        xt = torch.where(noise, torch.randint_like(x0, self.vocab_size), x0)
        sel = torch.randint(num_canvases, (B,), device=dev)

        # Encoder: plain causal next-token loss on the clean sequence.
        full = torch.cat([prompt, x0], 1)
        enc_logits = self.lm(input_ids=full).logits[:, :-1]
        enc_loss = F.cross_entropy(enc_logits.flatten(0, 1).float(), full[:, 1:].flatten())

        # Decoder: first pass without self-conditioning (no grad), then feed its logits
        # back for a random half of the batch.
        with torch.no_grad():
            first, _ = self._decode_selected(prompt, x0, xt, sel, None, canvas_size)
        use_sc = torch.rand(B, device=dev) < self_cond_prob
        sc_logits = torch.where(use_sc[:, None, None], first, torch.zeros_like(first))
        logits, idx = self._decode_selected(prompt, x0, xt, sel, sc_logits, canvas_size)
        target = x0.gather(1, idx - prompt.shape[1])
        dec_loss = F.cross_entropy(logits.flatten(0, 1).float(), target.flatten())
        return enc_loss, dec_loss

    # ---------------------------------------------------------------- sampling

    def _denoise_step(self, canvas, sc_emb, cache, ctx_len):
        """One decoder forward: canvas attends to the cached context and to itself."""
        B, C = canvas.shape
        emb = self.self_conditioner(self.embed(canvas), sc_emb)
        mask = torch.zeros(B, 1, C, ctx_len + C, dtype=emb.dtype, device=emb.device)
        out = self.lm.base_model(
            inputs_embeds=emb, attention_mask=mask, past_key_values=copy.deepcopy(cache)
        )
        return self.head(out.last_hidden_state)

    @torch.no_grad()
    def generate(
        self,
        prompt,
        num_canvases,
        canvas_size=256,
        max_steps=48,
        entropy_bound=0.1,
        max_temp=0.8,
        min_temp=0.4,
        early_stop=True,
        entropy_threshold=0.005,
    ):
        """Block-autoregressive diffusion sampling.
        Returns (tokens [B, K*C], steps per canvas, committed fraction per canvas), where
        committed = share of the canvas the model had accepted on its own before the final
        step force-accepts everything (1.0 if it stopped early)."""
        B, P = prompt.shape
        cache = self.lm.base_model(input_ids=prompt, use_cache=True).past_key_values
        out, steps_used, committed = [], [], []
        for k in range(num_canvases):
            ctx_len = P + k * canvas_size
            canvas = torch.randint(self.vocab_size, (B, canvas_size), device=prompt.device)
            sc_emb = torch.zeros_like(self.embed(canvas))
            done = torch.zeros(B, dtype=torch.bool, device=prompt.device)
            for step in range(max_steps):
                noise = 1.0 - step / max_steps
                temp = min_temp + noise * (max_temp - min_temp)
                logits = self._denoise_step(canvas, sc_emb, cache, ctx_len).float() / temp
                sampled, accept = sample_from_predictions(
                    logits, entropy_bound, step == max_steps - 1, self.vocab_size
                )
                if step < max_steps - 1:
                    frac = accept.float().mean().item()
                if early_stop:  # ChainedEarlyStop(TokenStability, Entropy)
                    stable = (logits.argmax(-1) == canvas).all(-1)
                    confident = _entropy(logits).mean(-1) <= entropy_threshold
                    stop = stable & confident
                else:
                    stop = torch.zeros_like(done)
                canvas = torch.where(done[:, None], canvas, sampled)
                done = done | stop
                sc_emb = self.encode_logits(logits)
                if done.all():
                    break
            steps_used.append(step + 1)
            committed.append(1.0 if done.all() else frac)
            # Commit the finished canvas to the context with a causal encoder pass.
            cache = self.lm.base_model(
                input_ids=canvas, past_key_values=cache, use_cache=True
            ).past_key_values
            out.append(canvas)
        return torch.cat(out, 1), steps_used, committed


@torch.no_grad()
def check_attention_support(model, prompt_len=5, canvas_size=4):
    """Fail loudly if the HF model ignores custom 4D masks (e.g. hard-coded causal bias).
    Checks: (1) canvas is bidirectional, (2) context never sees the canvas,
    (3) the KV-cached sampling path equals the full-sequence training path."""
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(0)
    V = model.vocab_size
    prompt = torch.randint(V, (1, prompt_len), generator=g).to(dev)
    x0 = torch.randint(V, (1, 2 * canvas_size), generator=g).to(dev)
    xt = torch.randint(V, (1, 2 * canvas_size), generator=g).to(dev)
    sel = torch.tensor([1], device=dev)
    base, _ = model._decode_selected(prompt, x0, xt, sel, None, canvas_size)

    xt2 = xt.clone()
    xt2[0, -1] = (xt2[0, -1] + 1) % V  # change the LAST token of the selected canvas
    changed, _ = model._decode_selected(prompt, x0, xt2, sel, None, canvas_size)
    if torch.allclose(base[0, 0], changed[0, 0], atol=1e-5):
        raise RuntimeError("decoder is not bidirectional: this model ignores 4D attention masks")

    ctx = torch.cat([prompt, x0[:, :canvas_size]], 1)
    cache = model.lm.base_model(input_ids=ctx, use_cache=True).past_key_values
    canvas = xt[:, canvas_size:]
    sc = torch.zeros_like(model.embed(canvas))
    cached = model._denoise_step(canvas, sc, cache, ctx.shape[1])
    if not torch.allclose(base.float(), cached.float(), atol=1e-3, rtol=1e-3):
        raise RuntimeError("cached sampling path disagrees with the training path")
