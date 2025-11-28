# Twitter Sentiment Analysis and Classification

This project focuses on performing sentiment analysis and classification on Twitter data. It was developed as part of the "Aprenentatge Computacional" (Computational Learning) course at Universitat Autònoma de Barcelona (UAB).

## Project Overview

The goal of this project is to build machine learning models capable of classifying tweets based on their sentiment (e.g., positive, negative, neutral) or other specific categories.

## Dataset

*   **Source:** The dataset used was found in hugging face (https://huggingface.co/datasets/bdstar/Tweets-Sentiment-Analysis) taking only the train split.
*   **Description:** The dataset contains more than 1.5 million tweets labeled with 'positive', 'neutral' or 'negative'.
*   **Preprocessing:** Steps taken to clean the data include:
    *   Removing URLs, mentions, and hashtags.
    *   Tokenization and lemmatization.
    *   Stopword removal.

## Methodology

The project explores several machine learning algorithms, including:

1.  **Feature Extraction:**
    *   Bag of Words (BoW)
    *   TF-IDF
    *   Word Embeddings (e.g., Word2Vec, GloVe)
2.  **Models:**
    *   Naïve Bayes
    *   Support Vector Machines (SVM)
    *   Logistic Regression
    *   [Any Deep Learning models if applicable, e.g., LSTM, BERT]

## Installation

To run this project locally, follow these steps:

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/Tweeter-Sentiment-Analysis-and-Classification.git
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the code 'IMPORTARDATASETGRANDE.PY' to get the initial dataset we used

## Usage

Run the main script 'logreg_tweets.py' to train the models and get the results from it. Also you will see some graphics to view the analisys in the directory 'GRAFIQUES'.


