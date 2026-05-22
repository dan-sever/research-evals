"""
Download Hugging Face datasets and save each one as a local Parquet file.

Run:
    python load-datasets.py
"""

import os
from datasets import load_dataset

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)


def _save(ds, filename):
    """Take the first available split and save it as parquet."""
    split_name = next(iter(ds))  # e.g. 'train', 'test', 'validation'
    df = ds[split_name].to_pandas()
    path = os.path.join(OUT_DIR, filename)
    df.to_parquet(path)
    print(f"Saved {len(df)} rows from '{split_name}' split -> {path}")


def save_finsearchcomp():
    """FinSearchComp — financial search benchmark."""
    ds = load_dataset("ByteSeedXpert/FinSearchComp")
    _save(ds, "finsearchcomp.parquet")


def save_sealqa_seal0():
    """SealQA seal-0 — main split."""
    ds = load_dataset("vtllms/sealqa", "seal_0")
    _save(ds, "sealqa_seal0.parquet")


def save_sealqa_seal_hard():
    """SealQA seal-hard — harder questions."""
    ds = load_dataset("vtllms/sealqa", "seal_hard")
    _save(ds, "sealqa_seal_hard.parquet")


def save_sealqa_longseal():
    """SealQA longseal — long-context variant."""
    ds = load_dataset("vtllms/sealqa", "longseal")
    _save(ds, "sealqa_longseal.parquet")


if __name__ == "__main__":
    save_finsearchcomp()
    save_sealqa_seal0()
    save_sealqa_seal_hard()
    save_sealqa_longseal()