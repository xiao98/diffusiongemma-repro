"""Stage-1 SFT: turn a pretrained causal LM into a block-diffusion LM.

Optimizer settings follow the official configs (sft_sudoku_full.py / sft_pubmedqa.py):
clip 1.0, Adam b1=0.95 b2=0.99, weight decay 1e-4, 100 warmup steps, cosine to 1e-5.
"""

import argparse
import json
import math
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from data import chunks
from dgemma import DiffusionLM, check_attention_support

p = argparse.ArgumentParser()
p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
p.add_argument("--init", choices=["pretrained", "random"], default="pretrained")
p.add_argument("--steps", type=int, default=5000)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--prompt-len", type=int, default=256)
p.add_argument("--canvas", type=int, default=256)
p.add_argument("--num-canvases", type=int, default=2)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--out", default="run")
args = p.parse_args()

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(args.model)
if args.init == "pretrained":
    lm = AutoModelForCausalLM.from_pretrained(args.model).float()  # v5 loads bf16 by default
else:
    lm = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(args.model)).float()
model = DiffusionLM(lm).to(dev)
check_attention_support(model.eval())
model.train()

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.95, 0.99), weight_decay=1e-4)
warmup, end_lr = 100, 1e-5


def lr_at(step):
    if step < warmup:
        return args.lr * step / warmup
    frac = (step - warmup) / max(1, args.steps - warmup)
    return end_lr + 0.5 * (args.lr - end_lr) * (1 + math.cos(math.pi * frac))


chunk_len = args.prompt_len + args.num_canvases * args.canvas
log = open(f"{args.out}.jsonl", "w")
t0 = time.time()
for step, batch in enumerate(chunks(tok, "train", chunk_len, args.batch, limit=args.steps)):
    batch = batch.to(dev)
    prompt, x0 = batch[:, : args.prompt_len], batch[:, args.prompt_len :]
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        enc_loss, dec_loss = model.sft_loss(prompt, x0, args.canvas)
    opt.zero_grad(set_to_none=True)
    (enc_loss + dec_loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 20 == 0 or step == args.steps - 1:
        rec = {
            "step": step,
            "enc_loss": round(enc_loss.item(), 4),
            "dec_loss": round(dec_loss.item(), 4),
            "tokens_seen": (step + 1) * args.batch * chunk_len,
            "sec": round(time.time() - t0, 1),
        }
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

torch.save({"state": model.state_dict(), "args": vars(args)}, f"{args.out}.pt")
print("saved", f"{args.out}.pt")
