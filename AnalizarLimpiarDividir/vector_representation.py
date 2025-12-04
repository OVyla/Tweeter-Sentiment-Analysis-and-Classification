import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# ---------------------------
# Config per als splits
# ---------------------------
INPUT_DIR = "DATASETS/SPLIT"
TRAIN_FILE = "twitter_trainBALANCED.csv"
VAL_FILE   = "twitter_valBALANCED.csv"
TEST_FILE  = "twitter_testBALANCED.csv"

# Carpeta per guardar les representacions
CACHE_DIR = "DATASETS/VECTORS"
# os.makedirs(CACHE_DIR, exist_ok=True)


def get_vectors(train_text, val_text, test_text, method='TFIDF'):
    """
    Dispatcher principal: selecciona l'estratègia segons el paràmetre 'method'.
    Methods: 'TFIDF' (defecte) o 'BOW'.
    """
    if method == 'BOW':
        return _get_bow(train_text, val_text, test_text)
    else:
        return _get_tfidf(train_text, val_text, test_text)

def _get_tfidf(train_text, val_text, test_text):
    print("Vectoritzant dades (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        max_features=80000, 
        ngram_range=(1,3), 
        sublinear_tf=True,
        min_df=2,
        max_df=0.95
    )
    X_train = vectorizer.fit_transform(train_text)
    X_val = vectorizer.transform(val_text)
    X_test = vectorizer.transform(test_text)
    return X_train, X_val, X_test, vectorizer

def _get_bow(train_text, val_text, test_text):
    print("Vectoritzant dades (Bag of Words)...")
    vectorizer = CountVectorizer(
        max_features=30000,
        ngram_range=(1,2),
        min_df=5,
        max_df=0.95
    )
    X_train = vectorizer.fit_transform(train_text)
    X_val = vectorizer.transform(val_text)
    X_test = vectorizer.transform(test_text)
    return X_train, X_val, X_test, vectorizer

def load_splits(
    input_dir: str = INPUT_DIR,
    train_file: str = TRAIN_FILE,
    val_file: str = VAL_FILE,
    test_file: str = TEST_FILE):
    """
    Llegeix els CSV de train, val i test des de input_dir
    i retorna tres DataFrames: train_df, val_df, test_df.
    """

    train_path = os.path.join(input_dir, train_file)
    val_path   = os.path.join(input_dir, val_file)
    test_path  = os.path.join(input_dir, test_file)

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    # Comprovem que hi ha 'text' i 'label'
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"El DataFrame {name} no té columna 'text' o 'label'.")

    return train_df, val_df, test_df

"""
# ---------------------------
# Sense cache: carregar + vectoritzar
# ---------------------------
def load_and_vectorize_splits(
    method: str = "TFIDF",
    input_dir: str = INPUT_DIR,
    train_file: str = TRAIN_FILE,
    val_file: str = VAL_FILE,
    test_file: str = TEST_FILE,
):
    "
    Llegeix els splits des de disc i retorna:
        train_df, val_df, test_df,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        vectorizer

    Tot llest per passar a qualsevol model.
    "
    train_df, val_df, test_df = load_splits(
        input_dir=input_dir,
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
    )

    X_train, X_val, X_test, vectorizer = get_vectors(
        train_df["text"].tolist(),
        val_df["text"].tolist(),
        test_df["text"].tolist(),
        method=method,
    )

    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    return (
        train_df,
        val_df,
        test_df,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        vectorizer,
    )"""

# ---------------------------
# Helper: path del fitxer de cache
# ---------------------------
def _get_cache_path(method: str) -> str:
    method = method.lower()
    filename = f"vectors_{method}.joblib"
    return os.path.join(CACHE_DIR, filename)

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import joblib  # per guardar/carregar objectes grans

# ---------------------------
# Config per als splits
# ---------------------------
INPUT_DIR = "DATASETS/SPLIT"
TRAIN_FILE = "twitter_trainBALANCED.csv"
VAL_FILE   = "twitter_valBALANCED.csv"
TEST_FILE  = "twitter_testBALANCED.csv"

# Carpeta per guardar les representacions
CACHE_DIR = "DATASETS/VECTORS"
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------
# Funció bàsica: vectorització
# ---------------------------
def get_vectors(train_texts, val_texts, test_texts, method="TFIDF"):
    """
    Vectoritza textos de train/val/test amb TFIDF o BOW i
    retorna X_train, X_val, X_test i el vectorizer entrenat.
    """
    method = method.upper()

    if method == "TFIDF":
        vectorizer = TfidfVectorizer(
            max_features=80000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
        )
    elif method == "BOW":
        vectorizer = CountVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
    else:
        raise ValueError(f"Mètode de vectorització desconegut: {method}")

    X_train = vectorizer.fit_transform(train_texts)
    X_val   = vectorizer.transform(val_texts)
    X_test  = vectorizer.transform(test_texts)

    return X_train, X_val, X_test, vectorizer


# ---------------------------
# Carregar splits (CSV)
# ---------------------------
def load_splits(
    input_dir: str = INPUT_DIR,
    train_file: str = TRAIN_FILE,
    val_file: str = VAL_FILE,
    test_file: str = TEST_FILE,
):
    """
    Llegeix els CSV de train, val i test des de input_dir
    i retorna tres DataFrames: train_df, val_df, test_df.
    """
    train_path = os.path.join(input_dir, train_file)
    val_path   = os.path.join(input_dir, val_file)
    test_path  = os.path.join(input_dir, test_file)

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"El DataFrame {name} no té columna 'text' o 'label'.")

    return train_df, val_df, test_df


# ---------------------------
# Sense cache: carregar + vectoritzar
# ---------------------------
"""def load_and_vectorize_splits(
    method: str = "TFIDF",
    input_dir: str = INPUT_DIR,
    train_file: str = TRAIN_FILE,
    val_file: str = VAL_FILE,
    test_file: str = TEST_FILE,
):
    "
    Llegeix els splits des de disc i retorna:
        train_df, val_df, test_df,
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        vectorizer
    "
    train_df, val_df, test_df = load_splits(
        input_dir=input_dir,
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
    )

    X_train, X_val, X_test, vectorizer = get_vectors(
        train_df["text"].tolist(),
        val_df["text"].tolist(),
        test_df["text"].tolist(),
        method=method,
    )

    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    return (
        train_df,
        val_df,
        test_df,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        vectorizer,
    )
"""

# ---------------------------
# Helper: path del fitxer de cache
# ---------------------------
def _get_cache_path(method: str) -> str:
    method = method.lower()
    filename = f"vectors_{method}.joblib"
    return os.path.join(CACHE_DIR, filename)


# ---------------------------
# NOVETAT: carregar + vectoritzar amb CACHE
# ---------------------------
def load_and_vectorize_splits(
    method: str = "TFIDF",# use_cache: bool = True,
    input_dir: str = INPUT_DIR,
    train_file: str = TRAIN_FILE,
    val_file: str = VAL_FILE,
    test_file: str = TEST_FILE,
):
    """
    Igual que load_and_vectorize_splits, però:

    - Si use_cache=True i existeix un fitxer de cache -> el carrega.
    - Si no existeix, calcula tot, ho guarda i després ho retorna.

    Retorna el mateix diccionari que podríem passar directament a models:
        {
            "train_df", "val_df", "test_df",
            "X_train", "X_val", "X_test",
            "y_train", "y_val", "y_test",
            "vectorizer"
        }
    """
    cache_path = _get_cache_path(method)

    if os.path.exists(cache_path):
        print(f"[CACHE] Carregant representació des de {cache_path}")
        data = joblib.load(cache_path)
        return data

    print(f"[CACHE] No trobat cache per {method}. Calculant representació...")
    train_df, val_df, test_df = load_splits(
        input_dir=input_dir,
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
    )

    X_train, X_val, X_test, vectorizer = get_vectors(
        train_df["text"].tolist(),
        val_df["text"].tolist(),
        test_df["text"].tolist(),
        method=method,
    )

    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values

    data = {
        "train_df":   train_df,
        "val_df":     val_df,
        "test_df":    test_df,
        "X_train":    X_train,
        "X_val":      X_val,
        "X_test":     X_test,
        "y_train":    y_train,
        "y_val":      y_val,
        "y_test":     y_test,
        "vectorizer": vectorizer,
    }

    if not os.path.exists(cache_path):
        print(f"[CACHE] Guardant representació a {cache_path}")
        joblib.dump(data, cache_path)

    return data