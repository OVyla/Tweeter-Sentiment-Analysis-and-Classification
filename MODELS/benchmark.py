import subprocess
import os

# --- CONFIGURATION ---
# The main output file for the benchmark results
OUTPUT_FILE = "benchmark.txt"

# List of models to benchmark.
# Each entry is a dictionary specifying the script to run and its working directory.
MODELS_TO_RUN = [
    {
        "name": "Logistic Regression",
        "script": "run_logistic.py",
        "cwd": "LogisticRegression"
    },
    {
        "name": "Support Vector Machine (SVM)",
        "script": "run_svm.py",
        "cwd": "SVM"
    },
    {
        "name": "Random Forest & Ensemble",
        "script": "run_random_forest.py",
        "cwd": "RandomForest"
    }
]

def main():
    """
    Orchestrates the execution of benchmarks for all configured models.
    It runs each model's script, captures its output, and saves it to a unified file.
    """
    print("--- STARTING COMPREHENSIVE MODEL BENCHMARK ---")
    
    # Get the absolute path to the MODELS directory, where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, '..', OUTPUT_FILE) # Save benchmark.txt in the project root

    # Initialize the output file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("============================================\n")
            f.write("==      COMPREHENSIVE BENCHMARK RERUN     ==\n")
            f.write("============================================\n\n")
    except IOError as e:
        print(f"Error: Could not write to output file {output_path}. Aborting. Details: {e}")
        return

    # --- Run each model's benchmark script ---
    for model in MODELS_TO_RUN:
        model_name = model["name"]
        script_path = os.path.join(base_dir, model["cwd"], model["script"])
        working_dir = os.path.join(base_dir, model["cwd"])

        print(f"\n--- Running: {model_name} ---")
        print(f"Executing: python {script_path}")

        if not os.path.exists(script_path):
            error_msg = f"ERROR: Script not found at {script_path}"
            print(error_msg)
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(f"\n--- BENCHMARK FAILED FOR: {model_name} ---\n")
                f.write(f"{error_msg}\n")
            continue

        try:
            # Execute the script as a subprocess
            process = subprocess.run(
                ['python3', model['script']],
                capture_output=True,
                text=True,
                cwd=working_dir,
                check=True  # This will raise a CalledProcessError if the script returns a non-zero exit code
            )
            
            # Append the captured stdout to the main output file
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(f"--- Results for {model_name} ---\n")
                f.write(process.stdout)
                f.write("\n\n")
            
            print(f"--- Finished: {model_name}. Results appended to {OUTPUT_FILE} ---")

        except FileNotFoundError:
            error_msg = f"ERROR: 'python3' command not found. Please ensure Python 3 is installed and in your PATH."
            print(error_msg)
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(f"\n--- BENCHMARK FAILED FOR: {model_name} ---\n")
                f.write(f"{error_msg}\n")

        except subprocess.CalledProcessError as e:
            # This catches errors within the script's execution
            error_msg = f"ERROR: The script {model['script']} failed with exit code {e.returncode}."
            print(error_msg)
            print("--- STDOUT ---")
            print(e.stdout)
            print("--- STDERR ---")
            print(e.stderr)
            
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(f"\n--- BENCHMARK FAILED FOR: {model_name} ---\n")
                f.write(f"Exit Code: {e.returncode}\n")
                f.write("--- STDOUT ---\n")
                f.write(e.stdout)
                f.write("--- STDERR --M-")
                f.write(e.stderr)
                f.write("\n\n")

        except Exception as e:
            # Catch any other unexpected errors
            error_msg = f"An unexpected error occurred while running {model['script']}: {e}"
            print(error_msg)
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(f"\n--- BENCHMARK FAILED FOR: {model_name} ---\n")
                f.write(f"{error_msg}\n")


    print("\n--- COMPREHENSIVE BENCHMARK COMPLETED ---")
    print(f"Final results are in: {output_path}")

if __name__ == "__main__":
    main()
