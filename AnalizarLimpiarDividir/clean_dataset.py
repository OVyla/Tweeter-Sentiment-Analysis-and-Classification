
import pandas as pd
import re

# --- Lematización ---
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Descargar recursos de nltk si no están
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()



# --- Cargar dataset ---
df = pd.read_csv(
    "twitter_balanced_train.csv"
)




def lemmatize_text(text):
    """
    Aplica lematización palabra a palabra.
    """
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])




def remove_duplicates(df: pd.DataFrame, subset_cols=['text']) -> pd.DataFrame:
    """Elimina duplicados"""
    df.drop_duplicates(subset=subset_cols, keep='first', inplace=True)
    return df


def lowercase_strip(text: str) -> str:
    """ Convierte el texto a minúsculas y
      elimina espacios al inicio y al final. """
    return text.lower().strip()

def remove_empty_texts(df, column='text'):
    """
    Remove rows where the text column is NaN or empty after stripping spaces.
    """
    # Remove NaNs
    df = df.dropna(subset=[column])
    # Remove empty strings
    df = df[df[column].str.strip() != '']
    df = df.reset_index(drop=True)
    return df

def remove_punctuation_space(text: str) -> str:
    """
    Elimina signos de puntuación y sustituye por espacios.
    """
    # Puntuación a eliminar: guiones, comas, puntos, signos de interrogación/exclamación
    PUNCTUATION = re.compile(r'[-,.!?;:…]+')
    # Sustituimos por espacio y convertimos a minúsculas
    return PUNCTUATION.sub(" ", text.lower())

def fix_abbr_en(x):
    """
    Expande abreviaciones comunes en inglés de tweets/mensajes.
    """
    if isinstance(x, list):
        words = x
    elif isinstance(x, str):
        words = x.split()
    else:
        raise TypeError('Input must be a string or a list of words.')

    abbrevs = {
        "u": "you",
        "r": "are",
        "ur": "your",
        "lol": "laughing out loud",
        "idk": "I do not know",
        "btw": "by the way",
        "omg": "oh my god",
        "thx": "thanks",
        "pls": "please",
        "plz": "please",
        "gr8": "great",
        "b4": "before",
        "l8r": "later",
        "imho": "in my humble opinion",
        "smh": "shaking my head",
        "tbh": "to be honest"
    }

    return " ".join([abbrevs[word.lower()] if word.lower() in abbrevs else word for word in words])

import re

def replace_links(text):
    # Patrón para detectar cualquier link común
    url_pattern = r'(http[s]?://\S+|www\.\S+|\S+\.ly/\S+)'
    return re.sub(url_pattern, '{link}', text)



def remove_repeated_vowels(text):
    """
    Elimina vocales consecutivas repetidas en cada palabra de un texto.
    Ej: "holaaa" -> "hola", "greeeaaaat" -> "great"
    """
    return re.sub(r'([aeiou])\1+', r'\1', text, flags=re.IGNORECASE)


def normalize_laughs_en(text):
    """
    Normaliza risas escritas en inglés en un texto.
    Ejemplos:
        hahaha, hahahaha -> haha
        hehe, hehehe -> hehe
        hoho, hohoho -> hoho
        lmao, rofl -> lol 
    """
    words = text.split()
    normalized = []

    for word in words:
        w = word.lower()
        if 'ha' in w and w.count('h') + w.count('a') > 3:
            normalized.append('haha')
        elif 'he' in w and w.count('h') + w.count('e') > 3:
            normalized.append('hehe')
        elif 'ho' in w and w.count('h') + w.count('o') > 3:
            normalized.append('hoho')
        elif w in ['lmao', 'rofl']:
            normalized.append('lol')
        else:
            normalized.append(word)
    
    return " ".join(normalized)


def remove_hashtag_symbol(text):
    """
    Quita el símbolo # pero deja la palabra.
    Ejemplo: #HappyDay -> HappyDay
    """
    return " ".join([word[1:] if word.startswith("#") else word for word in text.split()])


def remove_mentions(text):
    """
    Sustituye menciones de usuario por {mention}.
    Ejemplo: @john -> {mention}
    """
    return " ".join(['{mention}' if word.startswith('@') else word for word in text.split()])


# Función para corregir palabras en inglés
from spellchecker import SpellChecker

spell = SpellChecker(language='en', distance=1)

def correcting_words_en(text):
    misspelled = spell.unknown(text.split())
    return " ".join([
        (spell.correction(word) if word in misspelled else word) or word
        for word in text.split()
    ])


def remove_currency(text):
    """
    Replaces mentions of money symbols or currency words with {money}.
    Detects: €, $, £, yen, pound, euro, dollar
    """
    currency_words = ['€', '$', '£', 'yen', 'pound', 'euro', 'dollar']
    wlist = ['{money}' if any(c in word.lower() for c in currency_words) else word for word in text.split()]
    return " ".join(wlist)


def remove_special_characters(text):
    # Mantener solo letras, números y espacios
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)





def remove_obvious_spam(df, column='text', max_unique_words=2, min_length=20):
    """
    Remove obvious spam: texts with too few unique words and above a minimum length.
    
    Parameters:
    - df: DataFrame
    - column: name of the text column
    - max_unique_words: maximum number of unique words allowed (texts with <= this number are spam)
    - min_length: minimum text length to consider (texts shorter than this are ignored)
    """
    df['is_spam'] = df[column].apply(
        lambda t: 1 if (len(set(t.split())) <= max_unique_words and len(t) > min_length) else 0
    )
    df = df[df['is_spam'] == 0]
    df = df.drop(columns=['is_spam'])
    df = df.reset_index(drop=True)
    return df





# --- Aplicar pipeline de limpieza ---


df = remove_empty_texts(df)
df['text'] = df['text'].apply(lowercase_strip)
df['text'] = df['text'].apply(remove_punctuation_space)
df['text'] = df['text'].apply(fix_abbr_en)
df['text'] = df['text'].apply(replace_links)
df['text'] = df['text'].apply(remove_repeated_vowels)
df['text'] = df['text'].apply(normalize_laughs_en)
df['text'] = df['text'].apply(remove_hashtag_symbol)
df['text'] = df['text'].apply(remove_mentions)
# df['text'] = df['text'].apply(correcting_words_en)  # ¡Ojo! Esto puede ser lento en datasets grandes
df['text'] = df['text'].apply(remove_currency)
df['text'] = df['text'].apply(remove_special_characters)
# --- Lematización ---
df['text'] = df['text'].apply(lemmatize_text)
df = remove_obvious_spam(df)
df = remove_empty_texts(df)
df = remove_duplicates(df)

# --- Guardar dataset limpio ---
df.to_csv("twitter_balancedCLEAN.csv", index=False)
