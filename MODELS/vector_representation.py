import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectors(train_text, val_text, test_text):
    """
    Aplica TF-IDF amb els paràmetres optimitzats.
    Retorna les matrius disperses per train, val i test.
    """
    print("Vectoritzant dades (TF-IDF)...")
    
    vectorizer = TfidfVectorizer(
        max_features=80000, 
        ngram_range=(1,3), 
        sublinear_tf=True,
        min_df=2,
        max_df=0.95
    )
    
    # Ajustem només amb train, transformem la resta
    X_train = vectorizer.fit_transform(train_text)
    X_val = vectorizer.transform(val_text)
    X_test = vectorizer.transform(test_text)
    
    return X_train, X_val, X_test, vectorizer

def BoW(train_text, val_text, test_text):
    """
    Aplica Bag of Words (CountVectorizer) amb paràmetres similars a TF-IDF.
    Retorna les matrius disperses per train, val i test, i el vectorizer.
    """
    from sklearn.feature_extraction.text import CountVectorizer
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
