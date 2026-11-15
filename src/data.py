import pandas as pd
from src.config import RAW_DIR, PROCESSED_DIR

PROCESSED_FILES = {"train": "train.csv", "val": "validation.csv", "test": "test.csv"}


def read_lines(path):
    with open(path, encoding="utf-8", newline="\n") as f:
        return [line.rstrip("\r\n") for line in f]


def load_split(name):
    texts = read_lines(RAW_DIR / f"{name}_text.txt")
    labels = read_lines(RAW_DIR / f"{name}_labels.txt")
    assert len(texts) == len(labels), f"{name} : {len(texts)} textes, {len(labels)} labels"
    return pd.DataFrame({"text": texts, "label": [int(x) for x in labels]})


def drop_corrupted_rows(df):
    return df[~df["text"].str.contains("\t")].reset_index(drop=True)


def load_processed(name):
    return pd.read_csv(PROCESSED_DIR / PROCESSED_FILES[name])