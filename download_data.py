import os
import requests
import zipfile
import io

DATA_URL = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
TARGET_FILE = "training.1600000.processed.noemoticon.csv"

def download_dataset():
    if os.path.exists(TARGET_FILE):
        print(f"✅ El fitxer '{TARGET_FILE}' ja existeix.")
        return

    print("⬇️ Descarregant dataset (80MB+)...")
    response = requests.get(DATA_URL)
    
    if response.status_code == 200:
        print("📦 Descomprimint...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Extreu només el fitxer que necessitem
            z.extract(TARGET_FILE, path=".")
        print("✅ Fet! Dataset llest per fer servir.")
    else:
        print(f"❌ Error en la descàrrega: {response.status_code}")

if __name__ == "__main__":
    download_dataset()