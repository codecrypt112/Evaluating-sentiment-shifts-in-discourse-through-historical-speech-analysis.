#!/usr/bin/env python3
# train.py - Process historical speeches and train sentiment analysis models

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

from keras.utils import pad_sequences
import gensim
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel, pipeline
from bertopic import BERTopic
import warnings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Download necessary NLTK resources
def download_nltk_resources():
    """Download required NLTK resources."""
    resources = ['vader_lexicon', 'punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource)

# Data processing functions
def load_and_preprocess_data(file_path):
    """Load and preprocess the historical speeches data."""
    logger.info(f"Loading data from {file_path}")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert date strings to datetime objects
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Extract year and decade for analysis
    df['year'] = df['date'].dt.year
    df['decade'] = (df['year'] // 10) * 10
    
    # Basic text preprocessing
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    return df

def preprocess_text(text):
    """Clean and preprocess text for NLP tasks."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, keeping only letters, numbers, and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_texts(texts):
    """Tokenize a list of texts."""
    stop_words = set(stopwords.words('english'))
    tokenized_texts = []
    
    for text in texts:
        tokens = word_tokenize(text)
        # Remove stopwords and short tokens
        filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        tokenized_texts.append(filtered_tokens)
    
    return tokenized_texts

# VADER Sentiment Analysis
def analyze_vader_sentiment(df):
    """Analyze sentiment using VADER."""
    logger.info("Running VADER sentiment analysis")
    
    # Initialize the VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Apply VADER to get sentiment scores
    df['vader_scores'] = df['processed_text'].apply(sid.polarity_scores)
    
    # Extract individual sentiment components
    df['vader_neg'] = df['vader_scores'].apply(lambda x: x['neg'])
    df['vader_neu'] = df['vader_scores'].apply(lambda x: x['neu'])
    df['vader_pos'] = df['vader_scores'].apply(lambda x: x['pos'])
    df['vader_compound'] = df['vader_scores'].apply(lambda x: x['compound'])
    
    # Create a categorical sentiment label based on compound score
    df['vader_sentiment'] = df['vader_compound'].apply(
        lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
    )
    
    # Save the sentiment results for later use
    sentiment_results = df[['speaker', 'date', 'year', 'decade', 
                            'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 'vader_sentiment']]
    
    return df, sentiment_results

# LSTM Model for Sentiment Analysis
def build_lstm_model(df):
    """Build and train an LSTM model for sentiment classification."""
    logger.info("Building LSTM sentiment model")
    
    # Convert sentiment to numeric labels (negative=0, neutral=1, positive=2)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df['vader_sentiment'])
    
    # Save the label encoder for later use
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Tokenize texts
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['processed_text'])
    
    # Save tokenizer for later use
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(df['processed_text'])
    
    # Pad sequences to ensure same length
    max_length = 100
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, encoded_labels, test_size=0.2, random_state=42
    )
    
    # Build LSTM model
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 64
    
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        LSTM(units=64, dropout=0.2, recurrent_dropout=0.2),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')  # 3 classes: negative, neutral, positive
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    logger.info(f"LSTM Model - Test Accuracy: {accuracy:.4f}")
    
    # Save model
    model.save('models/lstm_sentiment_model.h5')
    
    # Return LSTM evaluation metrics and history for visualization
    return {
        'accuracy': accuracy,
        'loss': loss,
        'history': history.history,
        'validation_accuracy': max(history.history['val_accuracy'])
    }

# Topic Modeling with BERTopic
def perform_topic_modeling(df):
    """Perform topic modeling using BERTopic."""
    logger.info("Performing topic modeling with BERTopic")
    
    # Initialize BERTopic model
    topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
    
    # Fit the model on the processed texts
    topics, probs = topic_model.fit_transform(df['processed_text'])
    
    # Add topic assignments to the DataFrame
    df['topic'] = topics
    
    # Get topic information
    topic_info = topic_model.get_topic_info()
    
    # Save the topic model
    topic_model.save("models/bertopic_model")
    
    # Create a DataFrame with topic assignments
    topic_results = df[['speaker', 'date', 'year', 'decade', 'topic']]
    
    return df, topic_model, topic_info, topic_results

# Word Embedding Analysis
def analyze_word_embeddings(df):
    """Analyze word embeddings using Word2Vec."""
    logger.info("Creating Word2Vec embeddings")
    
    # Tokenize the processed texts
    tokenized_texts = tokenize_texts(df['processed_text'])
    
    # Train Word2Vec model
    w2v_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4
    )
    
    # Save the model
    w2v_model.save("models/word2vec_model.model")
    
    return w2v_model

# Historical sentiment trends analysis
def analyze_sentiment_trends(sentiment_results):
    """Analyze sentiment trends over time."""
    logger.info("Analyzing sentiment trends")
    
    # Aggregate sentiment by decade
    decade_sentiment = sentiment_results.groupby('decade')[
        ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']
    ].mean().reset_index()
    
    # Aggregate sentiment by speaker
    speaker_sentiment = sentiment_results.groupby('speaker')[
        ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']
    ].mean().reset_index()
    
    # Sort speakers by compound sentiment
    speaker_sentiment = speaker_sentiment.sort_values('vader_compound', ascending=False)
    
    # Save trend data for visualization
    decade_sentiment.to_csv('data/decade_sentiment.csv', index=False)
    speaker_sentiment.to_csv('data/speaker_sentiment.csv', index=False)
    
    return decade_sentiment, speaker_sentiment

# Main execution function
def main():
    """Main execution function."""
    logger.info("Starting historical speeches sentiment analysis")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Load and preprocess data
    df = load_and_preprocess_data('historical_speeches.csv')
    
    # Save processed data
    df.to_csv('data/processed_speeches.csv', index=False)
    
    # Analyze sentiment with VADER
    df, sentiment_results = analyze_vader_sentiment(df)
    sentiment_results.to_csv('data/sentiment_results.csv', index=False)
    
    # Build LSTM model
    lstm_metrics = build_lstm_model(df)
    
    # Perform topic modeling
    df, topic_model, topic_info, topic_results = perform_topic_modeling(df)
    topic_info.to_csv('data/topic_info.csv', index=False)
    topic_results.to_csv('data/topic_results.csv', index=False)
    
    # Analyze word embeddings
    w2v_model = analyze_word_embeddings(df)
    
    # Analyze sentiment trends
    decade_sentiment, speaker_sentiment = analyze_sentiment_trends(sentiment_results)
    
    logger.info("Analysis complete. Results saved to data directory.")
    
    # Return all results for potential further analysis
    return {
        'df': df,
        'sentiment_results': sentiment_results,
        'lstm_metrics': lstm_metrics,
        'topic_model': topic_model,
        'topic_info': topic_info,
        'topic_results': topic_results,
        'w2v_model': w2v_model,
        'decade_sentiment': decade_sentiment,
        'speaker_sentiment': speaker_sentiment
    }

if __name__ == "__main__":
    main()