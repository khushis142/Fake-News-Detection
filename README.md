# Fake News Detection & Classification System

An end-to-end Natural Language Processing (NLP) text analysis pipeline designed to automate the classification of unstructured news articles into real or fake categories. This project leverages dense semantic word embeddings via Word2Vec and robust machine learning classifiers to achieve high-accuracy text classification.

## Features

* **Advanced Text Preprocessing:** Full pipeline utilizing NLTK for automated tokenization, stopword removal, and morphological lemmatization.
* **Semantic Embeddings:** Implements a Word2Vec framework (via Gensim) to capture rich contextual and semantic relationships across documents, outperforming traditional Bag-of-Words (BoW) approaches.
* **Supervised Machine Learning:** Trains, benchmarks, and optimizes Logistic Regression and Random Forest classification frameworks to evaluate classification robustness.

## Tech Stack

* **Language:** Python
* **NLP & Embeddings:** NLTK, Gensim (Word2Vec)
* **Machine Learning:** Scikit-learn (Logistic Regression, Random Forest)
* **Data Manipulation:** Pandas, NumPy

## Architecture & Pipeline

1. **Data Preprocessing:** Input text undergoes clean-up where non-alphabetic characters are filtered out, text is lowercased, standard English stopwords are removed, and words are lemmatized to their base forms.
2. **Vectorization:** Tokenized documents are converted into dense vector representations by averaging the trained Word2Vec word vectors across each document.
3. **Classification:** Optimized machine learning models ingest the document vectors to perform binary classification (Real vs. Fake).

## Dataset

Kaggle: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data
