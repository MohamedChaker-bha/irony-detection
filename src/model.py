from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.config import TFIDF_PARAMS


def build_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf", clf), #classification model
    ])