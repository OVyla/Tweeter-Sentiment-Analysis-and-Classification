import run

# Configuració del Benchmark
MODELS = ['standard', 'ovo', 'ovr', 'grid']
VECTORS = ['TFIDF', 'BOW']
OUTPUT_FILE = "benchmark_output.txt"

def main():
    print("INICIANT BENCHMARK COMPLET...")
    
    # 1. Netejar/Inicialitzar el fitxer de sortida
    with open(OUTPUT_FILE, 'w') as f:
        f.write("BENCHMARK DE MODELS DE CLASSIFICACIÓ\n")
        f.write("====================================\n")

    # 2. Iterar sobre totes les combinacions
    count = 1
    total = len(MODELS) * len(VECTORS)

    for vec_method in VECTORS:
        for model_name in MODELS:
            print(f"[{count}/{total}] Processant...")
            
            # Cridem la pipeline amb mode 'a' (append) per no esborrar l'anterior
            run.run_pipeline(
                selected_model=model_name,
                vector_method=vec_method,
                output_file=OUTPUT_FILE,
                write_mode='a'
            )
            count += 1

    print(f"\nBenchmark finalitzat! Consulta els resultats a: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()