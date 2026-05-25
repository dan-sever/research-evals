"""
Download Hugging Face datasets and save each one as a local Parquet file.

Run:
    python load-datasets.py
"""

import os
from datasets import load_dataset

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)


def _save(load_fn, filename):
    """Download via `load_fn()` only if the parquet file does not already exist."""
    path = os.path.join(OUT_DIR, filename)
    if os.path.exists(path):
        print(f"Skipping {filename} (already exists). Delete the file to force re-download.")
        return
    ds = load_fn()
    split_name = next(iter(ds))  # e.g. 'train', 'test', 'validation'
    df = ds[split_name].to_pandas()
    df.to_parquet(path)
    print(f"Saved {len(df)} rows from '{split_name}' split -> {path}")


def save_finsearchcomp():
    """FinSearchComp — financial search benchmark."""
    _save(lambda: load_dataset("ByteSeedXpert/FinSearchComp"), "finsearchcomp.parquet")


def save_sealqa_seal0():
    """SealQA seal-0 — main split."""
    _save(lambda: load_dataset("vtllms/sealqa", "seal_0"), "sealqa_seal0.parquet")


def save_sealqa_seal_hard():
    """SealQA seal-hard — harder questions."""
    _save(lambda: load_dataset("vtllms/sealqa", "seal_hard"), "sealqa_seal_hard.parquet")


def save_sealqa_longseal():
    """SealQA longseal — long-context variant."""
    _save(lambda: load_dataset("vtllms/sealqa", "longseal"), "sealqa_longseal.parquet")


def save_deepsearchqa():
    """DeepSearchQA — Google's deep-search QA benchmark."""
    _save(lambda: load_dataset("google/deepsearchqa"), "deepsearchqa.parquet")


if __name__ == "__main__":
    save_finsearchcomp()
    save_sealqa_seal0()
    save_sealqa_seal_hard()
    save_sealqa_longseal()
    save_deepsearchqa()