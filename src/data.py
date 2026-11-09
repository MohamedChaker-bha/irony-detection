"""Loading the TweetEval irony splits."""
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, SPLITS


def download_raw() -> None:
    """Download TweetEval 'irony' from Hugging Face and save each split to data/raw/."""
    from datasets import load_dataset

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("tweet_eval", "irony")
    for split in SPLITS:
        ds[split].to_pandas().to_csv(RAW_DIR / f"{split}.csv", index=False)


def load_split(split: str = "train", processed: bool = True) -> pd.DataFrame:
    """Return one split ("train", "validation" or "test") as a DataFrame."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    folder = PROCESSED_DIR if processed else RAW_DIR
    return pd.read_csv(folder / f"{split}.csv")
