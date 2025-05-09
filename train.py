#!/usr/bin/env python3
"""
train.py - Sentiment Analysis of Historical Speeches

This script processes historical speech data, performs sentiment analysis using multiple
models (VADER, BERT, LSTM), and creates topic models. Results are saved for use in
the Streamlit dashboard.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from bertopic import BERTopic
import gensim
from gensim import corpora
import plotly.express as px
import plotly.graph_objects as go
import warnings
import json

warnings.filterwarnings('ignore')
np.random.seed(42)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# Download necessary NLTK resources
print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Load spaCy model
print("Loading spaCy model...")
try:
    nlp = spacy.load('en_core_web_sm')
except:
    print("Downloading spaCy model...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

def load_data(file_path='historical_speeches.csv'):
    """Load and prepare the dataset"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert date strings to datetime objects
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract year, decade, and century
    df['year'] = df['date'].dt.year
    df['decade'] = (df['year'] // 10) * 10
    df['century'] = ((df['year'] - 1) // 100) + 1
    
    # Sort by date
    df = df.sort_values('date')
    
    print(f"Loaded {len(df)} speeches from {df['year'].min()} to {df['year'].max()}")
    return df

def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def analyze_vader_sentiment(texts):
    """Analyze sentiment using VADER"""
    print("Analyzing sentiment with VADER...")
    analyzer = SentimentIntensityAnalyzer()
    
    sentiments = []
    for text in texts:
        scores = analyzer.polarity_scores(text)
        sentiments.append({
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        })
    
    return pd.DataFrame(sentiments)

def analyze_bert_sentiment(texts):
    """Analyze sentiment using BERT"""
    print("Analyzing sentiment with BERT...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    
    # Process in batches to avoid memory issues
    batch_size = 16
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_results = sentiment_pipeline(batch_texts)
        results.extend(batch_results)
    
    # Convert to DataFrame format
    sentiments = []
    for result in results:
        label = result['label']
        score = result['score']
        
        if label == 'POSITIVE':
            sentiments.append({
                'positive': score,
                'negative': 1 - score,
                'bert_compound': score * 2 - 1  # Scale to [-1, 1] like VADER
            })
        else:
            sentiments.append({
                'positive': 1 - score,
                'negative': score,
                'bert_compound': (1 - score) * 2 - 1  # Scale to [-1, 1] like VADER
            })
    
    return pd.DataFrame(sentiments)

def train_lstm_model(texts, labels):
    """Train LSTM model for sentiment prediction"""
    print("Training LSTM model...")
    
    # Tokenize texts
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)
    
    # Save tokenizer
    with open('models/lstm_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    max_length = 100
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, labels, test_size=0.2, random_state=42
    )
    
    # Build model
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100
    
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train model
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping]
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"LSTM Model - Test Accuracy: {accuracy:.4f}")
    
    # Save model
    model.save('models/lstm_sentiment_model.h5')
    
    # Save training history
    with open('models/lstm_training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    return model, tokenizer, history

def create_topic_model(texts, preprocessed_texts):
    """Create topic models using BERTopic"""
    print("Creating topic models with BERTopic...")
    
    # BERTopic model
    topic_model = BERTopic(nr_topics=10)
    topics, probs = topic_model.fit_transform(texts)
    
    # Save the model
    topic_model.save('models/bertopic_model')
    
    # Create gensim LDA model as well
    print("Creating LDA topic model...")
    tokenized_texts = [text.split() for text in preprocessed_texts]
    dictionary = corpora.Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    # Train LDA model
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=10,
        alpha='auto',
        passes=10,
        random_state=42
    )
    
    # Save the model
    lda_model.save('models/lda_model')
    dictionary.save('models/lda_dictionary')
    
    # Save corpus
    with open('models/lda_corpus.pkl', 'wb') as f:
        pickle.dump(corpus, f)
    
    return topic_model, lda_model, dictionary, corpus, topics

def extract_named_entities(texts):
    """Extract and analyze named entities using spaCy"""
    print("Extracting named entities...")
    entities = []
    
    for text in texts:
        doc = nlp(text)
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'type': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
    
    entities_df = pd.DataFrame(entities)
    
    # Save entities data
    entities_df.to_csv('data/named_entities.csv', index=False)
    
    return entities_df

def create_visualizations(df, vader_df, bert_df, topics, entities_df):
    """Create visualizations for the dashboard"""
    print("Creating visualizations...")
    
    # Combine dataframes
    df_with_sentiment = df.copy()
    df_with_sentiment['vader_compound'] = vader_df['compound']
    df_with_sentiment['vader_positive'] = vader_df['positive']
    df_with_sentiment['vader_negative'] = vader_df['negative']
    df_with_sentiment['vader_neutral'] = vader_df['neutral']
    
    df_with_sentiment['bert_compound'] = bert_df['bert_compound']
    df_with_sentiment['bert_positive'] = bert_df['positive']
    df_with_sentiment['bert_negative'] = bert_df['negative']
    
    df_with_sentiment['topic'] = topics
    
    # Save enhanced dataset
    df_with_sentiment.to_csv('data/speeches_with_sentiment.csv', index=False)
    
    # Create time series of sentiment
    time_sentiment = df_with_sentiment.groupby('year')[
        ['vader_compound', 'vader_positive', 'vader_negative', 'bert_compound']
    ].mean().reset_index()
    
    # Save time series data
    time_sentiment.to_csv('data/time_sentiment.csv', index=False)
    
    # Create sentiment by speaker
    speaker_sentiment = df_with_sentiment.groupby('speaker')[
        ['vader_compound', 'bert_compound']
    ].mean().reset_index()
    
    # Save speaker sentiment data
    speaker_sentiment.to_csv('data/speaker_sentiment.csv', index=False)
    
    # Create sentiment by decade
    decade_sentiment = df_with_sentiment.groupby('decade')[
        ['vader_compound', 'vader_positive', 'vader_negative', 'bert_compound']
    ].mean().reset_index()
    
    # Save decade sentiment data
    decade_sentiment.to_csv('data/decade_sentiment.csv', index=False)
    
    # Create topic distribution over time
    topic_time = df_with_sentiment.groupby(['decade', 'topic']).size().reset_index(name='count')
    
    # Save topic time data
    topic_time.to_csv('data/topic_time.csv', index=False)
    
    # Entity frequency
    entity_counts = entities_df['type'].value_counts().reset_index()
    entity_counts.columns = ['entity_type', 'count']
    
    # Save entity counts
    entity_counts.to_csv('data/entity_counts.csv', index=False)
    
    # Save speeches text by decade for word cloud
    decade_texts = df_with_sentiment.groupby('decade')['text'].apply(lambda x: ' '.join(x)).reset_index()
    decade_texts.to_csv('data/decade_texts.csv', index=False)
    
    print("Visualizations created and saved.")
    
    return df_with_sentiment

def analyze_sentiment_shifts(df_with_sentiment):
    """Analyze sentiment shifts over time"""
    print("Analyzing sentiment shifts...")
    
    # Calculate rolling average sentiment
    time_df = df_with_sentiment.sort_values('date')
    time_df['rolling_vader'] = time_df['vader_compound'].rolling(window=10).mean()
    time_df['rolling_bert'] = time_df['bert_compound'].rolling(window=10).mean()
    
    # Detect major shifts in sentiment
    time_df['vader_shift'] = time_df['rolling_vader'].diff().abs()
    time_df['bert_shift'] = time_df['rolling_bert'].diff().abs()
    
    # Find significant sentiment shifts (top 10%)
    vader_threshold = time_df['vader_shift'].quantile(0.9)
    bert_threshold = time_df['bert_shift'].quantile(0.9)
    
    significant_shifts = time_df[
        (time_df['vader_shift'] > vader_threshold) | 
        (time_df['bert_shift'] > bert_threshold)
    ].copy()
    
    # Add contextual information
    significant_shifts = significant_shifts[['date', 'speaker', 'vader_shift', 'bert_shift']]
    
    # Save significant shifts
    significant_shifts.to_csv('data/significant_shifts.csv', index=False)
    
    # Calculate sentiment volatility by decade
    volatility = df_with_sentiment.groupby('decade')[['vader_compound', 'bert_compound']].std().reset_index()
    volatility.columns = ['decade', 'vader_volatility', 'bert_volatility']
    
    # Save volatility data
    volatility.to_csv('data/sentiment_volatility.csv', index=False)
    
    print("Sentiment shift analysis complete.")
    
    return significant_shifts, volatility

def save_metadata(df, vader_df, bert_df, entities_df, topics):
    """Save metadata about the analysis for the dashboard"""
    metadata = {
        'num_speeches': len(df),
        'date_range': [df['date'].min().strftime('%Y-%m-%d'), df['date'].max().strftime('%Y-%m-%d')],
        'num_speakers': df['speaker'].nunique(),
        'top_speakers': df['speaker'].value_counts().head(10).to_dict(),
        'avg_vader_compound': vader_df['compound'].mean(),
        'avg_bert_compound': bert_df['bert_compound'].mean(),
        'num_entities': len(entities_df),
        'entity_types': entities_df['type'].value_counts().to_dict(),
        'topic_distribution': pd.Series(topics).value_counts().to_dict(),
        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('data/metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    print("Metadata saved.")
    return metadata

def main():
    """Main function to execute the sentiment analysis pipeline"""
    print("Starting sentiment analysis pipeline...")
    
    # Load and prepare data
    df = load_data()
    
    # Preprocess text
    print("Preprocessing texts...")
    df['preprocessed_text'] = df['text'].apply(preprocess_text)
    
    # Analyze sentiment with VADER
    vader_df = analyze_vader_sentiment(df['text'].tolist())
    
    # Analyze sentiment with BERT
    bert_df = analyze_bert_sentiment(df['text'].tolist())
    
    # Create binary labels for LSTM (using VADER compound score)
    binary_labels = (vader_df['compound'] > 0).astype(int).values
    
    # Train LSTM model
    lstm_model, tokenizer, history = train_lstm_model(
        df['preprocessed_text'].tolist(), binary_labels
    )
    
    # Create topic models
    topic_model, lda_model, dictionary, corpus, topics = create_topic_model(
        df['text'].tolist(), df['preprocessed_text'].tolist()
    )
    
    # Extract named entities
    entities_df = extract_named_entities(df['text'].tolist())
    
    # Create visualizations
    df_with_sentiment = create_visualizations(df, vader_df, bert_df, topics, entities_df)
    
    # Analyze sentiment shifts
    significant_shifts, volatility = analyze_sentiment_shifts(df_with_sentiment)
    
    # Save metadata
    metadata = save_metadata(df, vader_df, bert_df, entities_df, topics)
    
    print("Sentiment analysis pipeline completed successfully!")
    print(f"Results saved in 'data/', 'models/', and 'visualizations/' directories.")

if __name__ == "__main__":
    main()