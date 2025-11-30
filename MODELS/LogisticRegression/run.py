import run
from tqdm import tqdm  # Importem la barra de progrés

# Configuració
MODELS = ['standard', 'ovo', 'ovr', 'grid']
VECTORS = ['TFIDF', 'BOW']
OUTPUT_FILE = "benchmark_output.txt"

def main():
    print("INICIANT BENCHMARK OPTIMITZAT...")
    
    # Inicialitzar fitxer
    with open(OUTPUT_FILE, 'w') as f:
        f.write("BENCHMARK DE MODELS DE CLASSIFICACIÓ\n")
        f.write("====================================\n")

    # Calculem el total de tasques per configurar la barra
    total_steps = len(VECTORS) * len(MODELS)

    # Iniciem la barra de progrés
    with tqdm(total=total_steps, desc="Progrés Total", unit="model") as pbar:
        
        for vec_method in VECTORS:
            # Actualitzem la descripció de la barra
            pbar.set_description(f"Preparant dades ({vec_method})...")
            
            # 1. Preparació de dades (es fa un cop per vectorització)
            data_packet = run.prepare_data(vec_method)
            
            for model_name in MODELS:
                # Actualitzem descripció per saber què està entrenant ara mateix
                pbar.set_description(f"Entrenant {model_name} amb {vec_method}")
                
                # 2. Entrenament i report
                run.train_and_report(
                    data=data_packet,
                    model_name=model_name,
                    vector_method=vec_method,
                    output_file=OUTPUT_FILE,
                    write_mode='a'
                )
                
                # Avancem un pas la barra
                pbar.update(1)

    print(f"\nBenchmark finalitzat! Consulta: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()