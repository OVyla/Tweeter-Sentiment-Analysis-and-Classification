# Twitter Sentiment Analysis and Classification

This project, developed for the "Aprenentatge Computacional" (Computational Learning) course at Universitat Autònoma de Barcelona (UAB), performs sentiment analysis and classification on Twitter data. It explores various machine learning models to classify tweets into 'positive', 'neutral', or 'negative' categories.

## Project Structure

The repository is organized as follows:

```
.
├── AnalizarLimpiarDividir/ # Scripts for data processing
│   ├── Analizardataset.py
│   ├── clean_dataset.py
│   ├── IMPORTARDATASETGRANDE.py
│   └── SplitDataset.py
├── DATASETS/                 # Raw and processed data
│   ├── twitter_balancedCLEAN.csv
│   └── ...
├── GRAFIQUES/                # Generated plots and visualizations
│   ├── analisisDataset/
│   ├── logreg/
│   └── WORDCLOUD/
├── MODELS/                   # ML model implementations
│   ├── LogisticRegression/
│   ├── RandomForest/
│   └── SVM/
├── PLOTS/                    # Benchmark plots for model comparison
│   ├── benchmark_accuracy.png
│   ├── benchmark_f1.png
│   └── benchmark_time.png
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Dataset

*   **Source**: The dataset is sourced from [Hugging Face Datasets](https://huggingface.co/datasets/bdstar/Tweets-Sentiment-Analysis) (train split).
*   **Description**: It contains over 1.5 million tweets, each labeled with 'positive', 'neutral', or 'negative' sentiment.
*   **Preprocessing**: The data cleaning process (`clean_dataset.py`) involves:
    *   Lowercasing text.
    *   Removing URLs, user mentions, and hashtags.
    *   Lemmatization using NLTK's WordNet.
    *   Removal of common English stopwords.

## Getting Started

Follow these instructions to set up and run the project locally.

### 1. Prerequisites

*   Python 3.8+
*   Git

### 2. Installation

Clone the repository to your local machine:
```bash
git clone https://github.com/your-username/Tweeter-Sentiment-Analysis-and-Classification.git
cd Tweeter-Sentiment-Analysis-and-Classification
```

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

Download the NLTK data required for preprocessing:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```

### 3. Data Preparation

First, download the original dataset by running the following script. This will fetch the data from Hugging Face and save it in the `DATASETS` directory.
```bash
python AnalizarLimpiarDividir/IMPORTARDATASETGRANDE.py
```

Next, clean and preprocess the dataset:
```bash
python AnalizarLimpiarDividir/clean_dataset.py
```

Finally, split the cleaned data into training, validation, and test sets:
```bash
python AnalizarLimpiarDividir/SplitDataset.py
```

## Usage

The project includes implementations for several machine learning models.

### Running Models

You can train and evaluate the models by running their respective `run` scripts. For example, to run the Logistic Regression model:
```bash
python MODELS/LogisticRegression/run.py
```

Similarly, for other models:
```bash
# Support Vector Machine
python MODELS/SVM/run_svm.py

# Random Forest
python MODELS/RandomForest/run_random_forest.py
```

### Benchmarking

The scripts generate performance metrics and save them in files like `analisis_resultats.txt` and `benchmark_output.txt`. The `PLOTS` directory contains visualizations comparing the accuracy, F1-score, and execution time of the different models.

## Models Explored

*   **Logistic Regression**: Implemented with different vector representations (Bag-of-Words, TF-IDF) and multiclass strategies (One-vs-Rest, One-vs-One).
*   **Support Vector Machines (SVM)**: Both `LinearSVC` and `SVC` with different kernels are tested.
*   **Ensemble Methods**:
    *   Random Forest
    *   AdaBoost
    *   Gradient Boosting
    *   LightGBM

Results, including confusion matrices, ROC curves, and validation curves, are saved in the `GRAFIQUES` directory.