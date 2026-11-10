"""Project-wide constants: paths, random seed, column and label names."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42

TEXT_COL = "text_clean"
RAW_TEXT_COL = "text"
LABEL_COL = "label"

LABEL_NAMES = {0: "not_irony", 1: "irony"}
SPLITS = ("train", "validation", "test")