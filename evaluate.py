"""Compare diffusion decoding vs autoregressive decoding of the SAME fine-tuned weights.

Reports the paper's quantities at small scale:
  tokens/sec, denoising steps per canvas, tokens per forward (TPF),
  and generation quality as perplexity under the frozen ORIGINAL pretrained model.
"""

import argparse
import json
import time

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from data import chunks
from dgemma import DiffusionLM

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--batches", type=int, default=8)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--no-early-stop", action="store_true")
p.add_argument("--ar-temp", type=float, default=1.0)
p.add_argument("--entropy-bound", type=float, default=0.1)  # official sampler default
p.add_argument("--max-steps", type=int, default=48)  # paper sweeps 48..192 for the SFT model
ev = p.parse_args()

ck = torch.load(ev.ckpt, map_location="cpu")
a = ck["args"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a["model"])
model = DiffusionLM(AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(a["model"])).float())
model.load_state_dict(ck["state"])
model.to(dev).eval()
judge = AutoModelForCausalLM.from_pretrained(a["model"]).float().to(dev).eval()

P, C, K = a["prompt_len"], a["canvas"], a["num_canvases"]
new_tokens = C * K


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def judge_nll(prompt, cont):
    """Mean NLL of the continuation under the frozen pretrained model."""
    full = torch.cat([prompt, cont], 1)
    logits = judge(input_ids=full).logits[:, P - 1 : -1]
    return F.cross_entropy(logits.flatten(0, 1).float(), cont.flatten()).item()


def distinct2(x):
    bigrams = torch.stack([x[:, :-1], x[:, 1:]], -1)
    return sum(len({tuple(b) for b in row.tolist()}) for row in bigrams) / bigrams[..., 0].numel()


stats = {k: {"nll": [], "sec": 0.0, "d2": []} for k in ("real", "diffusion", "ar")}
steps, committed = [], []
with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
    for batch in chunks(tok, "validation", P + new_tokens, ev.batch, limit=ev.batches):
        batch = batch.to(dev)
        prompt, real = batch[:, :P], batch[:, P:]

        sync(); t = time.time()
        diff, used, comm = model.generate(
            prompt, K, C, early_stop=not ev.no_early_stop, entropy_bound=ev.entropy_bound,
            max_steps=ev.max_steps,
        )
        committed += comm
        sync(); stats["diffusion"]["sec"] += time.time() - t
        steps += used

        sync(); t = time.time()
        ar = model.lm.generate(
            prompt, attention_mask=torch.ones_like(prompt), do_sample=True,
            temperature=ev.ar_temp, top_k=0, max_new_tokens=new_tokens,
            min_new_tokens=new_tokens, pad_token_id=tok.eos_token_id,
        )[:, P:]
        sync(); stats["ar"]["sec"] += time.time() - t

        for name, cont in (("real", real), ("diffusion", diff), ("ar", ar)):
            stats[name]["nll"].append(judge_nll(prompt, cont))
            stats[name]["d2"].append(distinct2(cont))

n_tok = ev.batches * ev.batch * new_tokens
mean = lambda xs: sum(xs) / len(xs)
report = {
    "ckpt": ev.ckpt,
    "canvas": C,
    "avg_denoising_steps_per_canvas": round(mean(steps), 2),
    "tokens_per_forward": round(C / mean(steps), 2),
    "entropy_bound": ev.entropy_bound,
    "max_steps": ev.max_steps,
    "ar_temp": ev.ar_temp,
    "committed_before_final_step": round(mean(committed), 3),
}
for name, s in stats.items():
    report[name] = {
        "judge_ppl": round(float(torch.tensor(mean(s["nll"])).exp()), 2),
        "distinct2": round(mean(s["d2"]), 3),
    }
    if name != "real":
        report[name]["tokens_per_sec"] = round(n_tok / s["sec"], 1)
report["speedup_vs_ar"] = round(
    report["diffusion"]["tokens_per_sec"] / report["ar"]["tokens_per_sec"], 2
)
print(json.dumps(report, indent=2))
print("\n--- diffusion sample ---\n" + tok.decode(diff[0][:200]))
print("\n--- AR sample ---\n" + tok.decode(ar[0][:200]))
json.dump(report, open(ev.ckpt.replace(".pt", f".eval.eb{ev.entropy_bound}.n{ev.max_steps}.json"), "w"), indent=2)
