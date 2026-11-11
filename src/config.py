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

# Final TF-IDF settings chosen in notebook 02 (validation F1 macro)
TFIDF_PARAMS = dict(
    token_pattern=r"[\w']+|[^\w\s]",
    lowercase=False,
    ngram_range=(1, 2),
    min_df=2,
)