from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

RANDOM_STATE = 42
TEXT_COL = "text_clean"
LABEL_NAMES = ["non_irony", "irony"]

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"

# Placeholder: replace with the winner of your vectorizer comparison.
TFIDF_PARAMS = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
