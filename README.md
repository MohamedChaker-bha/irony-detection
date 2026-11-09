# Irony Detection

Binary irony detection on tweets using the **TweetEval `irony`** dataset
(0 = not ironic, 1 = ironic). We compare classical ML models with a fine-tuned transformer.

## Project structure

```
data/raw/          original TweetEval files, never edited          (A)
data/processed/    train/val/test CSVs with text_clean             (A)
src/               shared code imported by all notebooks
  config.py        paths, RANDOM_STATE, TEXT_COL, label names      (B)
  data.py          load_split("train") -> DataFrame                (A)
  cleaning.py      clean_text(text) -> cleaned text                (A)
  evaluate.py      evaluate(...) and save_results(...)             (B)
notebooks/
  01_eda_cleaning.ipynb   (A)
  02_nb_logreg.ipynb      (B)
  03_svm_rf_knn.ipynb     (C)
  04_transformer.ipynb    (D)
  05_comparison.ipynb     (together)
results/           metrics_<X>.csv per member + figures/*.png
models/            saved models (git-ignored)
final/             the ONE notebook we submit (+ HTML export)
```

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order (`01` → `05`). Notebooks import shared code with:

```python
import sys; sys.path.append("..")
from src.config import *
from src.data import load_split
```
