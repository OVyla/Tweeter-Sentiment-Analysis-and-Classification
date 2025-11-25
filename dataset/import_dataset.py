from datasets import load_dataset
import pandas as pd
import os

# ------------------------------
# 1️⃣ CardiffNLP TweetEval: solo neutrales
# ------------------------------
tweet_eval = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train")
tweet_eval_neutral = tweet_eval.filter(lambda x: x['label'] == 1)  # 1 = neutral

df_eval_neutral = pd.DataFrame({
    "text": tweet_eval_neutral['text'],
    "label": "neutral"
})

# ------------------------------
# 2️⃣ bdstar Twitter Sentiment: validation -> solo neutrales
# ------------------------------
bd_validation = load_dataset("bdstar/twitter-sentiment-analysis", split="validation")
bd_validation_neutral = bd_validation.filter(lambda x: x['label'] == 'neutral')

df_bd_validation_neutral = pd.DataFrame({
    "text": bd_validation_neutral['text'],
    "label": "neutral"
})

# ------------------------------
# 3️⃣ bdstar Twitter Sentiment: test -> solo positivos y negativos
# ------------------------------
bd_test = load_dataset("bdstar/twitter-sentiment-analysis", split="test")

# Filtrar positivos y negativos
bd_test_pos_neg = bd_test.filter(lambda x: x['label'] in ['positive', 'negative'])

# Submuestrear hasta 50,000 filas balanceadas
df_test_pos_neg = pd.DataFrame({
    "text": bd_test_pos_neg['text'],
    "label": bd_test_pos_neg['label']
})

# Contar cuántos hay de cada clase
min_count = 33220  # mitad y mitad para 50k
df_pos = df_test_pos_neg[df_test_pos_neg['label'] == 'positive'].sample(n=min_count, random_state=42)
df_neg = df_test_pos_neg[df_test_pos_neg['label'] == 'negative'].sample(n=min_count, random_state=42)

df_test_balanced = pd.concat([df_pos, df_neg], ignore_index=True)

# ------------------------------
# 4️⃣ Combinar todo
# ------------------------------
combined = pd.concat([df_eval_neutral, df_bd_validation_neutral, df_test_balanced], ignore_index=True)

# Mezclar aleatoriamente
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Limitar filas si quieres (opcional)
max_rows = 100000
if len(combined) > max_rows:
    combined = combined.sample(n=max_rows, random_state=42).reset_index(drop=True)

# ------------------------------
# 5️⃣ Revisar distribución
# ------------------------------
print("Distribución de clases:")
print(combined['label'].value_counts())

# ------------------------------
# 6️⃣ Guardar CSV
# ------------------------------
# Define your specific folder path here
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "twitter_combined.csv")

combined.to_csv(output_path, index=False)
print(f"Dataset combinado listo: {output_path}")
