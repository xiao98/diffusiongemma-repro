# DiffusionGemma stage-1, reproduced at small scale

A PyTorch port of the supervised fine-tuning stage and block sampler of
[DiffusionGemma](https://arxiv.org/abs/2608.00146), applied to a 135M causal LM so it
trains on one Colab A100. Written by reading the official JAX code, not the paper alone.

## Official source (cloned into `third_party/`, not committed)

```bash
git clone https://github.com/google/hackable_diffusion third_party/hackable_diffusion
git clone https://github.com/google-deepmind/gemma third_party/gemma   # see gemma/diffusion/
```

| This repo | Official file |
|---|---|
| `dgemma.py: sft_loss` | `gemma/diffusion/hackable_diffusion_adapter/hd/sft_model.py` |
| `dgemma.py: _decoder_mask` | `.../hd/mask_helpers.py: create_decoder_attention_mask` |
| `dgemma.py: SelfConditioning` | `gemma/diffusion/_transformer.py` |
| `dgemma.py: encode_logits` | `gemma/gm/nn/gemma4/_modules.py: Embedder.encode_logits` |
| `dgemma.py: sample_from_predictions`, `generate` | `gemma/diffusion/_sampler.py`, `_early_stopping.py`, `_chat_sampler.py` |
| `train.py` optimizer | `.../configs/sft_sudoku_full.py` |

## What is reproduced

- Uniform (multinomial) corruption with a linear schedule; no time conditioning.
- One set of weights as causal encoder + bidirectional canvas decoder.
- Loss = encoder next-token CE + unweighted denoising CE on one randomly selected canvas.
- Self-conditioning: two decoder passes, first without gradient, fed back with probability 0.5.
- Sampler: random initial canvas, at most 48 steps, temperature 0.8 -> 0.4, entropy-bounded
  token acceptance (bound 0.1) with uniform renoising, early stop when tokens are stable and
  mean entropy <= 0.005, finished canvas committed to the KV cache by a causal pass.

## Deviations

1. Base model is SmolLM2-135M (Llama architecture), not the Gemma 4 26B MoE. The official
   toolkit only loads the 26B checkpoint and asks for at least 2 A100s even with LoRA.
2. The official configs fine-tune the already-converted model on Sudoku/PubMedQA. Here the
   AR -> diffusion conversion itself is trained (paper stage 1), on TinyStories.
3. Stage 2 (RL + sampler distillation) is not reproduced; the paper does not specify the
   reward. The paper's SFT-only numbers (Fig. 9, Section 5) are the ones to compare
   against: with the default 48 steps the SFT model collapses into repetition and stops
   early; it needs up to 192 steps, where it reaches about 5 tokens per forward. The 20
   tokens per forward at 48 steps come from stage 2.
4. Training runs the decoder as a full-sequence forward with a 4D mask instead of a KV
   cache. `check_attention_support` asserts the two paths agree and runs before training.
5. Fixed-length text chunks: no padding, EOS-fill, stop-token truncation, sliding-window
   layers or multimodal inputs.
6. Self-conditioner uses PyTorch default init (official init not checked).
7. The self-conditioner's output norm is rescaled to the base model's embedding RMS.
   The official post-norm (RMS 1) suits Gemma's sqrt(d)-scaled embeddings; on SmolLM2 it
   inflated the decoder input 7.6x.

## Run

Local check (CPU): `python -m pytest -q test_dgemma.py`

Colab: run `python make_notebook.py`, open `colab.ipynb` in Colab, upload `dgemma-colab.zip`
when asked. It trains the warm-started model and evaluates it at several maximum step
counts, printing: denoising steps per canvas, tokens per forward, tokens/sec for diffusion
vs AR decoding of the same weights, and perplexity of the samples under the frozen
pretrained model. The random-init ablation is
`python train.py --init random --out scratch`.

Raw outputs of every run below are in `results/`. The port is derived from the official
Apache-2.0 code; see `LICENSE`.

`--model google/gemma-3-270m` should work after accepting its licence but is untested.

## Status

Unit tests pass.

Run 1 (A100, 3000 steps, before deviation 7 was fixed) is not a valid test of warm start:
diffusion samples scored judge perplexity 50.7 (warm) and 33.2 (random init) against 5.7
for real text, every canvas used all 48 steps, and samples were mostly high-frequency
tokens. Two suspected causes, both untested so far: the 7.6x input scale, and the
entropy bound of 0.1 being too strict for a weak model, so most of the canvas is only
decided at the forced final step. Run 2 reports `committed_before_final_step` and sweeps
the entropy bound to test the second one.

Run 2 (A100, 3000 steps, 36.9M tokens, after the fix). Judge perplexity of real text is
5.68 (distinct-2 0.756). Mean denoising loss over the last 500 steps: warm 1.82, random
init 2.28.

| init | entropy bound | committed before final step | diffusion ppl | distinct-2 | AR ppl (same weights) |
|---|---|---|---|---|---|
| warm | 0.1 | 0.16 | 57.2 | 0.485 | 7.7 |
| warm | 1.0 | 0.56 | 38.7 | 0.670 | 7.2 |
| warm | 5.0 | 1.00 | 19.7 | 0.678 | 7.1 |
| random | 0.1 | 0.19 | 71.4 | 0.528 | 69.7 |
| random | 1.0 | 0.52 | 62.4 | 0.678 | 68.0 |
| random | 5.0 | 0.91 | 67.0 | 0.702 | 68.8 |

- Warm start now beats random init on training loss and at every bound.
- The official bound of 0.1 is too strict for this model: 84% of the canvas is only
  decided at the forced final step. Loosening it takes warm from 57 to 20.
- The scale fix alone did not help at bound 0.1 (warm 50.7 -> 57.2), so the claim that the
  scale bug caused Run 1's warm < random result is not confirmed by these numbers.
- Diffusion decoding is still far from AR decoding of the same weights (19.7 vs 7.1), and
  early stopping almost never fires (47-48 of 48 steps), so tokens per forward stays at the
  256/48 cap and does not measure anything. The paper's ~5 TPF is not reproduced.
  (Superseded by Run 3: the paper's ~5 TPF is measured at N = 192, not 48.)
- The 3.5x tokens/sec over AR mostly reflects per-token Python overhead of HF `generate`
  on a small model, not a like-for-like speedup.

Run 3 (warm start retrained with the same seed, last-500-step denoising loss 1.82 as in
Run 2; sweep of the maximum number of denoising steps N, the axis of the paper's Fig. 9).
64 samples, one seed, no error bars.

| entropy bound | N | steps used | committed before final step | diffusion ppl | distinct-2 |
|---|---|---|---|---|---|
| 0.1 | 48 (Run 2) | 48 | 0.16 | 57.2 | 0.485 |
| 0.1 | 96 | 96 | 0.37 | 51.3 | 0.640 |
| 0.1 | 192 | 192 | 0.70 | 31.5 | 0.700 |
| 5.0 | 48 (Run 2) | 47.1 | 1.00 | 19.7 | 0.678 |
| 5.0 | 192 | 72.1 | 1.00 | 22.5 | 0.714 |

- At the official bound, quality improves with N (57 -> 51 -> 32), the direction the paper
  reports for its SFT checkpoint. It stays far from AR decoding of the same weights (6.9).
- At the official bound the model commits about one token per step at every N
  (41 tokens in 47 steps, 94 in 95, 180 in 191): the entropy budget only ever admits the
  one token the rule always accepts. The paper's SFT checkpoint at N = 192 stops after
  about 56 steps (read off Fig. 9), i.e. about 4.5 tokens per step. This model is much
  less confident; whether that is training budget, model size or the open-ended task is
  not determined here.
- With bound 5.0 adaptive stopping does fire at N = 192 (72 steps, 3.55 tokens per
  forward), but quality is no better than at N = 48 (22.5 vs 19.7, within what one seed
  can resolve).
- Paper Algorithm 1 returns the argmax at the last step; the released sampler samples.
  This port follows the code.
