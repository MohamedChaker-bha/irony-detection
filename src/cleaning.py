import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

NEGATIONS = {"no", "nor", "not", "never", "ain", "don", "didn", "doesn", "hadn",
             "hasn", "haven", "isn", "aren", "couldn", "shouldn", "wasn", "weren",
             "won", "wouldn", "mightn", "mustn", "needn", "shan"}

STOPS_ALL = set(stopwords.words("english"))
STOPS_KEEP_NEG = {w for w in STOPS_ALL if w not in NEGATIONS and "n't" not in w}

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

lemmatizer = WordNetLemmatizer()


def clean_text(text,
               remove_links=True,
               remove_mentions=True,
               hashtags="strip_symbol",
               lowercase=True,
               remove_stopwords=True,
               remove_punct=True,
               lemmatize=True):
    text = text.replace("\u2019", "'")

    if remove_links:
        text = URL_RE.sub(" ", text)
    if remove_mentions:
        text = MENTION_RE.sub(" ", text)
    if hashtags == "strip_symbol":
        text = text.replace("#", "")
    if lowercase:
        text = text.lower()

    words = text.split()

    if remove_stopwords:
        words = [w for w in words if w.lower().strip(string.punctuation) not in STOPS_KEEP_NEG]
    if remove_punct:
        words = [w.translate(PUNCT_TABLE) for w in words]
        words = [w for w in words if w != ""]
    if lemmatize:
        words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)