from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

RANDOM_STATE = 42
TEXT_COL = "text_clean"
LABEL_NAMES = ["non_irony", "irony"]