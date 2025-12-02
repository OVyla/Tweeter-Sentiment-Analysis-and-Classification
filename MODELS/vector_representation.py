import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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