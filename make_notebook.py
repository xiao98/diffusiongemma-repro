"""Writes colab.ipynb and the code zip to upload to Colab."""
import json, zipfile

FILES = ["dgemma.py", "data.py", "train.py", "evaluate.py", "test_dgemma.py"]
with zipfile.ZipFile("dgemma-colab.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        z.write(f)

STEPS = 3000
# Paper Fig. 9: the SFT-only model is evaluated with max steps N from 48 to 192.
EVALS = "\n".join(
    f"!python evaluate.py --ckpt warm.pt --entropy-bound {b} --max-steps {n}"
    for b, n in ((0.1, 96), (0.1, 192), (5.0, 192))
)
cells = [
    ("markdown", "# DiffusionGemma stage-1 reproduction (small scale)\nRuntime -> Change runtime type -> **A100 GPU**. Run cells top to bottom."),
    ("code", "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    ("code", "# Upload dgemma-colab.zip (built by make_notebook.py)\n!rm -f dgemma-colab*.zip\nfrom google.colab import files\nfiles.upload()\n!unzip -o -q dgemma-colab.zip && pip install -q -U transformers datasets"),
    ("code", "!python -m pytest -q test_dgemma.py"),
    ("markdown", "## 1. Warm start from the pretrained AR model (the paper's setting)"),
    ("code", f"# skipped if warm.pt from the previous run is still on this runtime\nimport os\nif not os.path.exists('warm.pt'):\n    !python train.py --init pretrained --steps {STEPS} --out warm"),
    ("markdown", "## 2. Sweep the maximum number of denoising steps (paper Fig. 9)"),
    ("code", EVALS),
    ("code", "# keep results\n!zip -q results.zip *.json *.jsonl\nfiles.download('results.zip')"),
]
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "cells": [
        {"cell_type": t, "metadata": {}, "source": src, **({"outputs": [], "execution_count": None} if t == "code" else {})}
        for t, src in cells
    ],
}
json.dump(nb, open("colab.ipynb", "w"), indent=1)
