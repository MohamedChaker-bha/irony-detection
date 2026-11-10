"""
Loading the TweetEval *irony* splits (owner: Member A).

TweetEval stores text and labels in separate files, one tweet / label per line:
    data/raw/train_text.txt  + train_labels.txt
    data/raw/val_text.txt    + val_labels.txt        <- file prefix is "val"
    data/raw/test_text.txt   + test_labels.txt

Team convention (src/config.py): the split is called "validation" everywhere in code,
and the processed files are data/processed/{train,validation,test}.csv
with columns: text, text_clean, label.

Usage:
    from src.data import load_split
    train = load_split("train")                      # processed: text, text_clean, label
    raw   = load_split("train", processed=False)     # original: text, label
"""
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, SPLITS

# Split name used in code -> file prefix used by TweetEval in data/raw/.
RAW_PREFIX = {"train": "train", "validation": "val", "test": "test"}

# Official sizes of the irony splits; used to detect a broken read.
EXPECTED_SIZES = {"train": 2862, "validation": 955, "test": 784}


def _read_lines(path: Path) -> list:
    """Return the lines of a UTF-8 file, splitting on "\\n" only (tweets may hold other separators)."""
    with open(path, "r", encoding="utf-8", newline="\n") as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":  # file ends with a newline
        lines.pop()
    return [line.rstrip("\r") for line in lines]


def load_mapping() -> dict:
    """Return the label mapping of mapping.txt, e.g. {0: 'non_irony', 1: 'irony'}."""
    pairs = (line.split()[:2] for line in _read_lines(RAW_DIR / "mapping.txt") if line.strip())
    return {int(idx): name for idx, name in pairs}


def load_split(split: str = "train", processed: bool = True) -> pd.DataFrame:
    """Return one split ("train", "validation" or "test") as a DataFrame.

    processed=True  -> data/processed/<split>.csv   (text, text_clean, label)
    processed=False -> data/raw/ original files      (text, label)
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    if processed:
        # keep_default_na=False: an empty text_clean must stay "" and not become NaN.
        df = pd.read_csv(PROCESSED_DIR / f"{split}.csv", keep_default_na=False)
        df["label"] = df["label"].astype(int)
        return df

    prefix = RAW_PREFIX[split]
    texts = _read_lines(RAW_DIR / f"{prefix}_text.txt")
    labels = _read_lines(RAW_DIR / f"{prefix}_labels.txt")
    if len(texts) != len(labels):
        raise ValueError(f"{split}: {len(texts)} texts but {len(labels)} labels (files misaligned)")
    if len(texts) != EXPECTED_SIZES[split]:
        raise ValueError(f"{split}: expected {EXPECTED_SIZES[split]} rows, got {len(texts)}")
    return pd.DataFrame({"text": texts, "label": [int(x) for x in labels]})