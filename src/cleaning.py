"""
src/cleaning.py  --  OWNER: Member A
====================================
clean_text(text, **switches) -> cleaned text.

Every cleaning step is an independent ON/OFF switch so that the cleaning EXPERIMENT in
notebook 01 can test them one at a time. The SAME function must be applied to train, val and test.

TweetEval already replaced real user names by the token "@user" and real links by "http".
"""

import re                                         # regular expressions for pattern removal
import string                                     # string.punctuation = all ASCII punctuation marks

# ---------------------------------------------------------------------------------------------
# Stopwords. NLTK's English list is preferred; if NLTK / its data is missing (e.g. no internet),
# we fall back to scikit-learn's list so the code never crashes.
# ---------------------------------------------------------------------------------------------
try:
    from nltk.corpus import stopwords as _nltk_sw         # NLTK stopword corpus
    STOPWORDS = set(_nltk_sw.words("english"))            # raises LookupError if data not downloaded
except Exception:                                         # NLTK not installed OR data missing
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS   # built into scikit-learn
    STOPWORDS = set(ENGLISH_STOP_WORDS)

# Negations: words that FLIP the meaning ("not good"). Default stopword lists contain them,
# which is dangerous for irony, so we can protect them with keep_negations=True.
NEGATIONS = {
    "no", "not", "nor", "never", "none", "nobody", "nothing", "nowhere", "neither", "cannot",
    "n't", "dont", "doesnt", "didnt", "isnt", "wasnt", "arent", "werent", "wont", "wouldnt",
    "couldnt", "shouldnt", "cant", "havent", "hasnt", "hadnt", "aint",
    "don't", "doesn't", "didn't", "isn't", "wasn't", "aren't", "weren't", "won't", "wouldn't",
    "couldn't", "shouldn't", "can't", "haven't", "hasn't", "hadn't", "ain't", "mustn't",
    "needn't", "shan't", "mightn't",
    "don", "doesn", "didn", "isn", "wasn", "aren", "weren", "won", "wouldn", "couldn",  # NLTK fragments
    "shouldn", "haven", "hasn", "hadn", "mustn", "needn", "shan", "mightn", "t",
}

# ---------------------------------------------------------------------------------------------
# Compiled regular expressions (compiled once = faster than recompiling for every tweet).
# ---------------------------------------------------------------------------------------------
URL_RE = re.compile(r"(?:https?://\S+|www\.\S+|\bhttps?\b)", flags=re.IGNORECASE)  # real links + TweetEval's "http"
MENTION_RE = re.compile(r"@\w+")                      # "@user", "@john_doe" ...
HASHTAG_RE = re.compile(r"#\w+")                      # "#blessed" (whole hashtag)
HASHTAG_SYMBOL_RE = re.compile(r"#(\w+)")             # captures the word after "#" so we can keep it
EMOJI_RE = re.compile(                                # common emoji / pictograph Unicode blocks
    "["
    "\U0001F300-\U0001FAFF"                           # symbols, pictographs, faces, objects
    "\U00002600-\U000027BF"                           # misc symbols + dingbats (hearts, stars, ...)
    "\U0001F000-\U0001F2FF"                           # mahjong, cards, enclosed characters
    "\U0001F1E6-\U0001F1FF"                           # flags
    "\u2B00-\u2BFF\u2190-\u21FF\uFE0F\u200D"          # arrows, variation selector, zero width joiner
    "]+",
    flags=re.UNICODE,
)
SPACES_RE = re.compile(r"\s+")                        # one or more whitespace characters
PUNCT_TABLE = str.maketrans("", "", string.punctuation)   # table that deletes every ASCII punctuation char


def _lemmatizer():
    """Create an NLTK WordNet lemmatizer, or return None if NLTK / WordNet data is unavailable."""
    try:
        from nltk.stem import WordNetLemmatizer       # import here so NLTK stays optional
        lem = WordNetLemmatizer()                     # create the lemmatizer object
        lem.lemmatize("tests")                        # force WordNet to load (raises LookupError if missing)
        return lem
    except Exception:
        return None                                   # caller will simply skip lemmatization


_LEM = _lemmatizer()                                  # built once at import time


def clean_text(
    text: str,
    remove_urls: bool = True,          # delete links ("http", "https://...")          -> noise, usually safe
    remove_mentions: bool = True,      # delete "@user"                                -> noise, usually safe
    hashtags: str = "keep",            # "keep" (#word) | "strip" (word) | "remove" (nothing)
    lowercase: bool = False,           # "GREAT" -> "great" (loses sarcastic shouting)
    remove_emojis: bool = False,       # delete emojis (they can be irony markers)
    remove_stopwords: bool = False,    # delete "the", "is", "and", ...
    keep_negations: bool = True,       # when removing stopwords, protect "not", "no", "never", ...
    remove_punctuation: bool = False,  # delete "!!!", "...", "?" (irony signals)
    lemmatize: bool = False,           # "running" -> "run" (needs NLTK WordNet; skipped otherwise)
) -> str:
    """Clean ONE tweet according to the switches and return the cleaned string."""
    if not isinstance(text, str):                         # defensive: None / NaN -> empty string
        return ""

    if remove_urls:                                       # step 1: links
        text = URL_RE.sub(" ", text)                      # replace each link by a space
    if remove_mentions:                                   # step 2: user mentions
        text = MENTION_RE.sub(" ", text)                  # replace each @user by a space

    if hashtags == "strip":                               # step 3a: keep the word, drop the "#"
        text = HASHTAG_SYMBOL_RE.sub(r"\1", text)         # "#blessed" -> "blessed"
    elif hashtags == "remove":                            # step 3b: drop the whole hashtag
        text = HASHTAG_RE.sub(" ", text)                  # "#blessed" -> ""
    elif hashtags != "keep":                              # any other value is a programming error
        raise ValueError("hashtags must be 'keep', 'strip' or 'remove'")

    if lowercase:                                         # step 4: lower case
        text = text.lower()

    if remove_emojis:                                     # step 5: emojis
        text = EMOJI_RE.sub(" ", text)

    if remove_stopwords or lemmatize:                     # steps 6-7 work word by word
        stop = STOPWORDS - NEGATIONS if keep_negations else STOPWORDS   # optionally protect negations
        tokens = []                                       # words that survive
        for tok in text.split():                          # split on whitespace
            core = tok.lower().strip(string.punctuation)  # word without surrounding punctuation ("great!!!" -> "great")
            if remove_stopwords and core in stop:         # is it a stopword?
                continue                                  # yes -> drop it
            if lemmatize and _LEM is not None and core:   # step 7: lemmatize (only if available)
                lemma = _LEM.lemmatize(core)              # "friends" -> "friend"
                tok = tok.lower().replace(core, lemma, 1) # swap the word inside the token, keep attached punctuation
            tokens.append(tok)                            # keep this token
        text = " ".join(tokens)                           # rebuild the sentence

    if remove_punctuation:                                # step 8: punctuation (after stopwords so "don't" is recognised)
        text = text.translate(PUNCT_TABLE)                # delete every ASCII punctuation character

    return SPACES_RE.sub(" ", text).strip()               # collapse multiple spaces and trim the ends