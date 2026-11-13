# Irony Detection in Tweets

## Overview

A binary text-classification system that predicts whether a tweet is **ironic** or **not ironic**, built on the TweetEval irony dataset (SemEval-2018 Task 3). We compare classical machine-learning models on TF-IDF features with a fine-tuned transformer, and analyze *why* irony is hard for models that only count words: ironic tweets often use positive words ("love", "great") to describe negative situations.

> 🚧 Work in progress — Machine Learning mini-project, CI1, Faculté des Sciences de Bizerte.

## Dataset

| Split | Tweets | Used for |
|---|---|---|
| train | 2,862 | Training the models |
| validation | 955 | Choosing settings, comparing models |
| test | 784 | One final evaluation, at the end |

- **Source:** [TweetEval](https://github.com/cardiffnlp/tweeteval) — `irony` subset, from SemEval-2018 Task 3 (Van Hee et al., 2018)
- **Labels:** `0` = non_irony, `1` = irony
- **Reference score:** the best system in the original SemEval-2018 competition reached **F1 = 0.71**

## Approach

1. **Exploratory data analysis** — class balance, tweet lengths, WordClouds per class, irony signals (punctuation, CAPS, emojis, leftover hashtags)
2. **Text cleaning experiment** — each cleaning step (lowercase, punctuation, stopwords, hashtags…) is tested on/off with the same baseline model, and kept only if it improves validation macro-F1
3. **Vectorization** — TF-IDF (word and character n-grams), always inside a scikit-learn `Pipeline` to avoid data leakage
4. **Classical models** — Naive Bayes, Logistic Regression, SVM, Random Forest, KNN, tuned with `GridSearchCV` (5-fold CV, macro-F1)
5. **Transformer** — fine-tuning a pre-trained transformer on the raw tweets, averaged over 3 random seeds
6. **Evaluation** — comparison on validation, a single final evaluation on test, and an error analysis of misclassified tweets

## Results

*To be filled once all models are evaluated.*

| Model | Validation macro-F1 | Test macro-F1 |
|---|---|---|
| Naive Bayes | — | — |
| Logistic Regression | — | — |
| SVM | — | — |
| Random Forest | — | — |
| KNN | — | — |
| Transformer | — | — |

## Key Takeaways

*To be written after the evaluation.*

## Tools & Libraries

- **Language:** Python 3.12
- **Data & plots:** pandas, NumPy, Matplotlib, seaborn, wordcloud
- **Classical ML & text:** scikit-learn, NLTK, joblib
- **Deep learning:** PyTorch, Hugging Face Transformers, Datasets, Evaluate, Accelerate
- **Workflow:** Jupyter, Git & GitHub

## Project Structure

```
data/raw/          original TweetEval files, never edited          (A)
data/processed/    train/validation/test with a text_clean column  (A)
src/               shared code imported by every notebook
  config.py        paths, RANDOM_STATE, column and label names     (B)
  data.py          load_split(...) -> DataFrame                    (A)
  cleaning.py      clean_text(...)                                 (A)
  evaluate.py      evaluate(...), save_results(...)                (B)
notebooks/
  01_eda_cleaning.ipynb    (A)
  02_nb_logreg.ipynb       (B)
  03_svm_rf_knn.ipynb      (C)
  04_transformer.ipynb     (D)
  05_comparison.ipynb      (together)
results/           metrics_<A|B|C|D>.csv + figures/
models/            saved models (not on GitHub — too large)
final/             the submitted notebook + its HTML export
```

## How to Run

**1. Clone the repository**

```bash
git clone https://github.com/MohamedChaker-bha/irony-detection.git
cd irony-detection
```

**2. Create an environment with Python 3.12** (check with `python --version`)

Option A — with conda:

```bash
conda create -n irony python=3.12
conda activate irony
```

Option B — without conda: install Python 3.12 from python.org, then in PyCharm create a *Virtualenv* interpreter based on Python 3.12.

**3. Install the shared libraries**

```bash
python -m pip install -r requirements.txt
```

**4. Task D only (transformer)** — install PyTorch first from [pytorch.org](https://pytorch.org) for your GPU, then:

```bash
python -m pip install -r requirements-transformer.txt
```

(See the comments in `requirements-transformer.txt` for the exact PyTorch version used.)

**5. Run the notebooks** in order, `01` → `05`. The submitted version is `final/irony_detection.ipynb`.

Notebooks import the shared code like this:

```python
import sys
sys.path.append("..")          # lets the notebook find the src/ folder

from src.config import RANDOM_STATE
from src.data import load_split
```

## Team Workflow

- Each member works on their own branch (`task-a`, `task-b`, `task-c`, `task-d`) and only edits the files they own.
- Work reaches `main` through Pull Requests.
- `requirements.txt` and `requirements-transformer.txt` are only changed on `main`.

## Authors

| Task | Member |
|---|---|
| A — Data, EDA & cleaning | *(name)* |
| B — Shared setup, Naive Bayes, Logistic Regression | **Sabrine Zouaoui** - [GitHub](https://github.com/sbrnn) - [LinkedIn] (https://www.linkedin.com/in/sabrine-zouaoui-4372903ba/) |
| C — SVM, Random Forest, KNN | *(name)* |
| D — Transformer fine-tuning | **Mohamed Chaker Iben Hadj Amor** — [GitHub](https://github.com/MohamedChaker-bha) · [LinkedIn](https://www.linkedin.com/in/mohamed-chaker-iben-hadj-amor) |

Course: Machine Learning — CI1, Faculté des Sciences de Bizerte. Supervisors: R. Mahouachi, Y. Haddad.
