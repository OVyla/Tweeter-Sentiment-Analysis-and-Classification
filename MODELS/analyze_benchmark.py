import re
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

def parse_benchmark_file(file_path):
    """
    Parses the benchmark output file and returns a list of dictionaries
    containing model performance metrics.
    """
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []

    current_model = None
    current_vectorizer = None
    training_time = None
    current_section = None # TRAIN, VALIDATION, TEST

    # Regex patterns
    model_pattern = re.compile(r"Model:\s+(.+?)\s+\|\s+Vectorització:\s+(.+)")
    time_pattern = re.compile(r"Temps entrenament:\s+([\d\.]+)\s+s")
    section_pattern = re.compile(r"---\s+(TRAIN|VALIDATION|TEST)\s+---")
    # Capture macro avg line: "macro avg     0.8042    0.8013    0.8020    136862" (after strip)
    macro_avg_pattern = re.compile(r"macro avg\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+[\d\.]+")
    accuracy_pattern = re.compile(r"Accuracy:\s+([\d\.]+)")

    metrics = {}

    for line in lines:
        line = line.strip()
        
        # 1. Detect Model & Vectorizer
        model_match = model_pattern.search(line)
        if model_match:
            # Save previous model data if complete (though we usually parse sequentially)
            current_model = model_match.group(1).strip()
            current_vectorizer = model_match.group(2).strip()
            metrics = {'Model': current_model, 'Vectorizer': current_vectorizer}
            continue

        # 2. Detect Training Time
        time_match = time_pattern.search(line)
        if time_match:
            training_time = float(time_match.group(1))
            metrics['Training Time (s)'] = training_time
            continue

        # 3. Detect Section (TRAIN/VAL/TEST)
        section_match = section_pattern.search(line)
        if section_match:
            current_section = section_match.group(1)
            continue

        # 4. Detect Accuracy
        acc_match = accuracy_pattern.search(line)
        if acc_match and current_section:
            metrics[f'{current_section} Accuracy'] = float(acc_match.group(1))
            continue

        # 5. Detect Macro Avg (Precision, Recall, F1)
        macro_match = macro_avg_pattern.search(line)
        if macro_match and current_section:
            # precision = float(macro_match.group(1))
            # recall = float(macro_match.group(2))
            f1 = float(macro_match.group(3))
            
            metrics[f'{current_section} Macro F1'] = f1
            
            # If we have parsed the TEST section, we assume this model block is done 
            # (or at least we have enough to save a row, but typically we wait or update)
            # The structure is Train -> Val -> Test. So after Test Macro F1, we are effectively done with this model instance.
            if current_section == 'TEST':
                 data.append(metrics.copy())
                 # Reset for safety, though the next "Model:" line will overwrite
            continue
            
    if not data:
        print("Debug: Parsed 0 records. Checking first few lines of file content for format mismatch:")
        for i in range(min(50, len(lines))):
             print(f"Line {i}: {lines[i].strip()}")


    return pd.DataFrame(data)

def plot_metrics(df, output_dir):
    """
    Generates and saves plots from the dataframe.
    """
    if df.empty:
        print("No data to plot.")
        return

    # Create a combined name for the x-axis
    df['Model_Vec'] = df['Model'] + " (" + df['Vectorizer'] + ")"

    # --- Plot 1: F1-Score Comparison ---
    plt.figure(figsize=(12, 6))
    
    # We'll plot Test Macro F1 primarily, maybe overlay Val?
    # Let's just do Test Macro F1 for simplicity and clarity as the primary benchmark.
    
    bars = plt.bar(df['Model_Vec'], df['TEST Macro F1'], color='skyblue', edgecolor='black')
    
    plt.title('Test Set Macro F1-Score by Model', fontsize=16)
    plt.xlabel('Model (Vectorization)', fontsize=12)
    plt.ylabel('Macro F1-Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.0)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_f1_score.png'))
    print(f"Saved benchmark_f1_score.png to {output_dir}")
    plt.close()

    # --- Plot 2: Training Time Comparison ---
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(df['Model_Vec'], df['Training Time (s)'], color='salmon', edgecolor='black')
    
    plt.title('Training Time by Model', fontsize=16)
    plt.xlabel('Model (Vectorization)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}s', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_training_time.png'))
    print(f"Saved benchmark_training_time.png to {output_dir}")
    plt.close()

    # --- Plot 3: Accuracy Comparison (Grouped) ---
    # If we want to compare Train vs Val vs Test Accuracy
    if 'TRAIN Accuracy' in df.columns and 'TEST Accuracy' in df.columns:
        plt.figure(figsize=(14, 6))
        import numpy as np
        
        x = np.arange(len(df))
        width = 0.25
        
        plt.bar(x - width, df['TRAIN Accuracy'], width, label='Train', color='#a8dadc')
        plt.bar(x, df['VALIDATION Accuracy'], width, label='Validation', color='#457b9d')
        plt.bar(x + width, df['TEST Accuracy'], width, label='Test', color='#1d3557')
        
        plt.title('Accuracy Comparison (Train vs Val vs Test)', fontsize=16)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(x, df['Model_Vec'], rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'benchmark_accuracy_grouped.png'))
        print(f"Saved benchmark_accuracy_grouped.png to {output_dir}")
        plt.close()

def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define output directory
    output_dir = os.path.join(current_dir, '..', 'GRAFIQUES', 'BENCHMARK')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    benchmark_file = os.path.join(current_dir, '..', 'benchmark.txt')
    
    print(f"Reading benchmark file from: {benchmark_file}")
    
    df = parse_benchmark_file(benchmark_file)
    
    if not df.empty:
        print("\nParsed Data:")
        print(df[['Model', 'Vectorizer', 'TEST Macro F1', 'Training Time (s)']])
        
        # Save CSV for reference
        csv_path = os.path.join(output_dir, 'benchmark_parsed_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSaved parsed data to {csv_path}")
        
        plot_metrics(df, output_dir)
    else:
        print("Could not parse any model data.")

if __name__ == "__main__":
    main()
