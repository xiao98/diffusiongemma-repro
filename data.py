"""TinyStories -> fixed-length token chunks [prompt | K canvases]."""

import torch
from datasets import load_dataset


def chunks(tokenizer, split, chunk_len, batch_size, limit=None):
    """Yields LongTensor [batch_size, chunk_len] of contiguous story text."""
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    buf, batch, n = [], [], 0
    eos = tokenizer.eos_token_id
    for row in ds:
        buf.extend(tokenizer(row["text"], add_special_tokens=False)["input_ids"] + [eos])
        while len(buf) >= chunk_len:
            batch.append(buf[:chunk_len])
            buf = buf[chunk_len:]
            if len(batch) == batch_size:
                yield torch.tensor(batch)
                batch = []
                n += 1
                if limit and n >= limit:
                    return
