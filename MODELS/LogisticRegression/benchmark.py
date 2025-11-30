import run

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

    # 1. Bucle exterior: Vectorització (es fa un cop per tipus)
    for vec_method in VECTORS:
        
        # AQUI està l'optimització: Calculem vectors UN SOL COP
        data_packet = run.prepare_data(vec_method)
        
        # 2. Bucle interior: Models (reutilitzen data_packet)
        for model_name in MODELS:
            run.train_and_report(
                data=data_packet,
                model_name=model_name,
                vector_method=vec_method,
                output_file=OUTPUT_FILE,
                write_mode='a'
            )

    print(f"\nBenchmark finalitzat! Consulta: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()