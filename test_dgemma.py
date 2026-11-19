import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from dgemma import DiffusionLM, check_attention_support, sample_from_predictions


def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=128)
    return DiffusionLM(AutoModelForCausalLM.from_config(cfg)).eval()


def test_masks_and_cache_equivalence():
    check_attention_support(tiny())


def test_context_rows_do_not_see_canvas():
    m = tiny()
    prompt = torch.randint(50, (1, 5))
    x0, xt = torch.randint(50, (1, 8)), torch.randint(50, (1, 8))
    allowed, _ = m._decoder_mask(torch.tensor([1]), 5, 4, 13)
    # rows of the prompt and of canvas 0 are strictly causal
    assert not allowed[0, 0, :9].triu(1)[:, :9].any()
    # rows of canvas 1 see everything up to the end of canvas 1
    assert allowed[0, 0, 9:13, :13].all()


def test_entropy_bound_selection():
    V = 6
    confident = torch.full((V,), -20.0); confident[2] = 20.0   # entropy ~ 0
    uniform = torch.zeros(V)                                   # entropy = ln 6
    logits = torch.stack([uniform, confident, uniform])[None]
    torch.manual_seed(0)
    hits = 0
    for _ in range(50):
        out, accept = sample_from_predictions(logits, 0.1, False, V)
        assert accept[0, 1] and accept.sum() == 2  # confident one + exactly one more
        assert out[0, 1] == 2          # confident token always accepted
        hits += int(out[0, 0] == out[0, 2])
    assert hits < 50                   # the others are random draws, not a fixed token
    # with bound 0.1: the confident token (entropy~0) is accepted, then the first uniform
    # one is also accepted because accumulated-before-it is still ~0; the second is not.


def test_overfits_one_batch():
    m = tiny().train()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    torch.manual_seed(1)
    prompt, x0 = torch.randint(50, (4, 6)), torch.randint(50, (4, 16))
    first = None
    for _ in range(150):
        enc, dec = m.sft_loss(prompt, x0, canvas_size=8)
        opt.zero_grad(); (enc + dec).backward(); opt.step()
        first = first or dec.item()
    assert dec.item() < 0.5 * first


def test_generate_shapes_and_early_stop():
    m = tiny()
    out, steps, committed = m.generate(
        torch.randint(50, (2, 5)), num_canvases=2, canvas_size=4, max_steps=6)
    assert out.shape == (2, 8) and len(steps) == 2 and all(1 <= s <= 6 for s in steps)
    assert all(0.0 <= c <= 1.0 for c in committed)


def test_decoder_input_keeps_base_embedding_scale():
    m = tiny()
    tokens = torch.randint(50, (1, 8))
    emb = m.embed(tokens)
    out = m.self_conditioner(emb, torch.zeros_like(emb))
    rms = lambda x: x.pow(2).mean().sqrt().item()
    assert abs(rms(out) / rms(m.lm.get_input_embeddings().weight) - 1) < 0.05
