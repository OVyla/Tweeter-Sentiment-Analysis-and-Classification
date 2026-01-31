# Twitter Sentiment Analysis

An advanced machine learning project for sentiment classification of tweets, featuring a comprehensive benchmarking of models and a modern web interface.

Developed for the "Aprenentatge Computacional" (Computational Learning) course at Universitat Autònoma de Barcelona (UAB).

![Web App Screenshot](https://via.placeholder.com/800x400?text=Web+App+Screenshot)

## 📌 Features

*   **Robust Preprocessing**: Custom cleaning pipeline including cleaning, lemmatization, and noise removal.
*   **Diverse Models**: Implementation and benchmarking of:
    *   Logistic Regression (TF-IDF & BoW)
    *   Support Vector Machines (SVM) - Linear & Kernel
    *   Naive Bayes (Multinomial, Bernoulli, Complement)
    *   Random Forest & Decision Trees
    *   Ensemble Methods (Stacking, Voting, LightGBM)
*   **Web Application**: A premium, responsive FastAPI web app to demonstrate the model in real-time.
*   **Bilingual Support**: The web interface supports both English and Catalan.

## 📂 Project Structure

```
.
├── app/                  # Web Application (FastAPI + Frontend)
├── data/                 # Datasets and cached vectors
├── models/               # Model implementations (SVM, LogReg, etc.)
├── plots/                # Generated visualizations and analysis
├── preprocessing/        # Data cleaning and splitting scripts
├── experiments/          # Experimental scripts
└── requirements.txt      # Dependencies
```

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   Git

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/Tweeter-Sentiment-Analysis-and-Classification.git
    cd Tweeter-Sentiment-Analysis-and-Classification
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data**:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    ```

## 🖥️ Running the Web App

To launch the interactive web interface:

1.  Ensure you have trained the models (or use the provided pre-trained ones).
2.  Run the application:
    ```bash
    uvicorn app.main:app --reload
    ```
3.  Open your browser and navigate to:
    `http://localhost:8000`

## 🧠 Training Models

To train the models from scratch:

```bash
# Logistic Regression Benchmark
python models/LogisticRegression/run_logistic_models.py

# SVM Benchmark
python models/SVM/run_svm.py
```

Results and plots will be generated in the `plots/` directory.

## 📊 Results

The project explores various vectorization techniques (TF-IDF vs Bag of Words) and model architectures. Detailed benchmarks and ROC curves can be found in the `plots/` directory.

---
*Created by Oriol*
