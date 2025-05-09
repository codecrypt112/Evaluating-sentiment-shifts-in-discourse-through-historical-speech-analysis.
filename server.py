#!/usr/bin/env python3
# server.py - Streamlit dashboard for historical speech sentiment analysis

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import pickle
import os
from datetime import datetime
import re
from bertopic import BERTopic
from gensim.models import Word2Vec
import networkx as nx
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

import spacy
from transformers import pipeline
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Historical Speech Sentiment Analysis",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stSidebar {
        background-color: #E8EAF6;
    }
    .stButton button {
        background-color: #3949AB;
        color: white;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sentiment-positive {
        color: #4CAF50;
    }
    .sentiment-negative {
        color: #F44336;
    }
    .sentiment-neutral {
        color: #9E9E9E;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_data():
    """Load preprocessed data."""
    try:
        processed_df = pd.read_csv('data/processed_speeches.csv')
        processed_df['date'] = pd.to_datetime(processed_df['date'])
        
        sentiment_df = pd.read_csv('data/sentiment_results.csv')
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        topic_df = pd.read_csv('data/topic_results.csv')
        topic_df['date'] = pd.to_datetime(topic_df['date'])
        
        topic_info = pd.read_csv('data/topic_info.csv')
        
        decade_sentiment = pd.read_csv('data/decade_sentiment.csv')
        speaker_sentiment = pd.read_csv('data/speaker_sentiment.csv')
        
        return {
            'processed': processed_df,
            'sentiment': sentiment_df,
            'topic': topic_df,
            'topic_info': topic_info,
            'decade_sentiment': decade_sentiment,
            'speaker_sentiment': speaker_sentiment
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_models():
    """Load trained models."""
    try:
        # Load VADER sentiment analyzer
        sid = SentimentIntensityAnalyzer()
        
        # Load LSTM model and tokenizer
        lstm_model = load_model('models/lstm_sentiment_model.h5')
        with open('models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load BERTopic model
        topic_model = BERTopic.load("models/bertopic_model")
        
        # Load Word2Vec model
        w2v_model = Word2Vec.load("models/word2vec_model.model")
        
        # Load Hugging Face transformer for advanced sentiment
        sentiment_pipeline = pipeline("sentiment-analysis")
        
        return {
            'vader': sid,
            'lstm': lstm_model,
            'tokenizer': tokenizer,
            'label_encoder': label_encoder,
            'topic_model': topic_model,
            'w2v_model': w2v_model,
            'transformer': sentiment_pipeline
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def preprocess_text(text):
    """Clean and preprocess text for analysis."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, keeping only letters, numbers, and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def analyze_new_speech(text, models):
    """Analyze a new speech text."""
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # VADER sentiment analysis
    vader_scores = models['vader'].polarity_scores(processed_text)
    
    # LSTM sentiment prediction
    sequence = models['tokenizer'].texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')
    lstm_prediction = models['lstm'].predict(padded_sequence)
    lstm_class = models['label_encoder'].inverse_transform([np.argmax(lstm_prediction[0])])[0]
    
    # Topic modeling
    topic, prob = models['topic_model'].transform([processed_text])
    topic_words = models['topic_model'].get_topic(topic[0])
    
    # Transformer-based sentiment
    transformer_result = models['transformer'](processed_text[:512])
    
    return {
        'vader': vader_scores,
        'lstm': {
            'prediction': lstm_prediction[0],
            'class': lstm_class
        },
        'topic': {
            'id': topic[0],
            'words': topic_words
        },
        'transformer': transformer_result[0]
    }

# Dashboard components
def render_header():
    """Render the dashboard header."""
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/8287/8287144.png", width=100)
    with col2:
        st.title("Historical Speech Sentiment Analysis")
        st.markdown("Analyze sentiment shifts in historical speeches over time")

def render_sidebar(data):
    """Render sidebar with filters and navigation."""
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard Overview", "Speaker Analysis", "Time Trends", 
         "Topic Modeling", "Sentiment Comparison", "New Speech Analysis"]
    )
    
    st.sidebar.header("Filters")
    
    # Time period filter
    min_year = int(data['processed']['year'].min())
    max_year = int(data['processed']['year'].max())
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    
    # Speaker filter
    all_speakers = sorted(data['processed']['speaker'].unique())
    selected_speakers = st.sidebar.multiselect(
        "Select Speakers",
        options=all_speakers,
        default=[]
    )
    
    # Apply filters
    filtered_data = {}
    for key, df in data.items():
        if key in ['processed', 'sentiment', 'topic']:
            temp_df = df.copy()
            # Filter by year range
            if 'year' in temp_df.columns:
                temp_df = temp_df[(temp_df['year'] >= year_range[0]) & 
                                 (temp_df['year'] <= year_range[1])]
            
            # Filter by selected speakers if any are selected
            if selected_speakers and 'speaker' in temp_df.columns:
                if selected_speakers:
                    temp_df = temp_df[temp_df['speaker'].isin(selected_speakers)]
            
            filtered_data[key] = temp_df
        else:
            filtered_data[key] = df
    
    return page, filtered_data

def render_dashboard_overview(data):
    """Render dashboard overview page."""
    st.header("Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Speeches", len(data['processed']))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Unique Speakers", len(data['processed']['speaker'].unique()))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Time Span", f"{data['processed']['year'].min()} - {data['processed']['year'].max()}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        positive_pct = (data['sentiment']['vader_sentiment'] == 'positive').mean() * 100
        st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Overall sentiment distribution
    st.subheader("Overall Sentiment Distribution")
    fig = px.pie(
        data['sentiment']['vader_sentiment'].value_counts().reset_index(),
        values='count',
        names='vader_sentiment',
        color='vader_sentiment',
        color_discrete_map={'positive': '#4CAF50', 'neutral': '#9E9E9E', 'negative': '#F44336'},
        hole=0.4
    )
    fig.update_layout(legend_title="Sentiment", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment across decades - MOVED INSIDE THE FUNCTION
    st.subheader("Sentiment Trends Across Decades")
    fig = px.line(
        data['decade_sentiment'],
        x='decade',
        y=['vader_pos', 'vader_neu', 'vader_neg'],
        markers=True,
        labels={'value': 'Sentiment Score', 'decade': 'Decade', 'variable': 'Sentiment Type'},
        color_discrete_map={
            'vader_pos': '#4CAF50',
            'vader_neu': '#9E9E9E',
            'vader_neg': '#F44336'
        }
    )
    fig.update_layout(legend_title="Sentiment Type", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent speeches - MOVED INSIDE THE FUNCTION
    st.subheader("Recent Speeches")
    recent = data['processed'].sort_values('date', ascending=False).head(5)
    for i, row in recent.iterrows():
        sentiment = data['sentiment'][data['sentiment']['date'] == row['date']].iloc[0]['vader_sentiment']
        sentiment_class = f"sentiment-{sentiment}"
        
        expander = st.expander(f"{row['speaker']} ({row['date'].strftime('%Y-%m-%d')})")
        with expander:
            st.markdown(f"**Sentiment:** <span class='{sentiment_class}'>{sentiment.capitalize()}</span>", 
                        unsafe_allow_html=True)
            st.write(row['text'][:300] + "..." if len(row['text']) > 300 else row['text'])

def render_sentiment_comparison(data):
    """Render sentiment comparison page."""
    st.header("Sentiment Analysis Comparison")
    
    # Compare VADER vs LSTM
    st.subheader("VADER vs LSTM Sentiment Classification")
    
    # Calculate agreement percentage
    # Note: This is simulated as we don't have LSTM predictions in our data
    st.info("This section simulates a comparison between VADER and LSTM sentiment classifications")
    
    agreement_percentage = 78.5  # Simulated value
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Agreement", f"{agreement_percentage:.1f}%")
    
    with col2:
        st.metric("VADER Positivity", 
                 f"{(data['sentiment']['vader_sentiment'] == 'positive').mean() * 100:.1f}%")
    
    # Confusion matrix (simulated)
    confusion = np.array([
        [25, 5, 2],   # VADER negative, LSTM [neg, neu, pos]
        [7, 30, 8],   # VADER neutral, LSTM [neg, neu, pos]
        [3, 10, 20]   # VADER positive, LSTM [neg, neu, pos]
    ])
    
    fig = px.imshow(
        confusion,
        x=['LSTM Negative', 'LSTM Neutral', 'LSTM Positive'],
        y=['VADER Negative', 'VADER Neutral', 'VADER Positive'],
        color_continuous_scale='Blues',
        labels=dict(x="LSTM Prediction", y="VADER Prediction", color="Count")
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Example: Sentiment Predictions on Sample Speeches
    st.subheader("Sample Speeches with Model Predictions")
    
    # Get a few sample speeches with different sentiments
    samples = data['sentiment'].groupby('vader_sentiment').apply(
        lambda x: x.sample(min(1, len(x)))
    ).reset_index(drop=True)
    
    for i, row in samples.iterrows():
        speech = data['processed'][
            (data['processed']['speaker'] == row['speaker']) & 
            (data['processed']['date'] == row['date'])
        ]
        
        if not speech.empty:
            speech = speech.iloc[0]
            expander = st.expander(f"{speech['speaker']} ({speech['date'].strftime('%Y-%m-%d')})")
            
            with expander:
                # VADER sentiment
                vader_sentiment = row['vader_sentiment']
                vader_class = f"sentiment-{vader_sentiment}"
                
                # LSTM sentiment (simulated)
                lstm_sentiments = ['negative', 'neutral', 'positive']
                lstm_sentiment = lstm_sentiments[np.random.randint(0, 3)]
                lstm_class = f"sentiment-{lstm_sentiment}"
                
                # Display text and predictions
                st.markdown(f"**VADER:** <span class='{vader_class}'>{vader_sentiment.capitalize()}</span> | " +
                           f"**LSTM:** <span class='{lstm_class}'>{lstm_sentiment.capitalize()}</span>", 
                           unsafe_allow_html=True)
                
                st.write(speech['text'][:300] + "..." if len(speech['text']) > 300 else speech['text'])
    
    # Sentiment distribution comparison
    st.subheader("Sentiment Distribution Comparison")
    
    # Simulated data for LSTM sentiment distribution
    lstm_dist = pd.DataFrame({
        'sentiment': ['negative', 'neutral', 'positive'],
        'VADER': [(data['sentiment']['vader_sentiment'] == 'negative').mean() * 100,
                 (data['sentiment']['vader_sentiment'] == 'neutral').mean() * 100,
                 (data['sentiment']['vader_sentiment'] == 'positive').mean() * 100],
        'LSTM': [20, 45, 35]  # Simulated values
    })
    
    lstm_dist_melted = lstm_dist.melt(id_vars='sentiment', var_name='model', value_name='percentage')
    
    fig = px.bar(
        lstm_dist_melted,
        x='sentiment',
        y='percentage',
        color='model',
        barmode='group',
        labels={'percentage': 'Percentage (%)', 'sentiment': 'Sentiment', 'model': 'Model'},
        color_discrete_map={'VADER': '#1E88E5', 'LSTM': '#FFC107'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def render_new_speech_analysis(models):
    """Render new speech analysis page."""
    st.header("New Speech Analysis")
    
    st.markdown("""
    Enter a new speech text to analyze its sentiment and topics. 
    The system will apply multiple sentiment analysis models and provide a comprehensive analysis.
    """)
    
    # Text input area
    new_text = st.text_area(
        "Enter speech text",
        height=200,
        placeholder="Enter the text of a speech to analyze..."
    )
    
    if st.button("Analyze Speech") and new_text:
        # Analyze the new speech
        analysis_results = analyze_new_speech(new_text, models)
        
        # Display results
        st.subheader("Analysis Results")
        
        # Sentiment scores
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### VADER Sentiment")
            vader_compound = analysis_results['vader']['compound']
            vader_sentiment = 'positive' if vader_compound >= 0.05 else ('negative' if vader_compound <= -0.05 else 'neutral')
            vader_class = f"sentiment-{vader_sentiment}"
            
            st.markdown(f"**Overall:** <span class='{vader_class}'>{vader_sentiment.capitalize()}</span>", 
                       unsafe_allow_html=True)
            st.metric("Compound Score", f"{vader_compound:.3f}")
            
            # Component scores
            components = {
                'Positive': analysis_results['vader']['pos'],
                'Neutral': analysis_results['vader']['neu'],
                'Negative': analysis_results['vader']['neg']
            }
            
            for label, score in components.items():
                st.text(f"{label}: {score:.3f}")
        
        with col2:
            st.markdown("### LSTM Sentiment")
            lstm_class = analysis_results['lstm']['class']
            lstm_class_css = f"sentiment-{lstm_class}"
            
            st.markdown(f"**Prediction:** <span class='{lstm_class_css}'>{lstm_class.capitalize()}</span>", 
                       unsafe_allow_html=True)
            
            # Class probabilities
            probs = analysis_results['lstm']['prediction']
            classes = ['negative', 'neutral', 'positive']
            
            for cls, prob in zip(classes, probs):
                st.text(f"{cls.capitalize()}: {prob:.3f}")
        
        with col3:
            st.markdown("### Transformer Sentiment")
            transformer_label = analysis_results['transformer']['label']
            transformer_score = analysis_results['transformer']['score']
            transformer_class = "sentiment-positive" if "positive" in transformer_label.lower() else "sentiment-negative"
            
            st.markdown(f"**Prediction:** <span class='{transformer_class}'>{transformer_label}</span>", 
                       unsafe_allow_html=True)
            st.metric("Confidence", f"{transformer_score:.3f}")
        
        # Topic analysis
        st.subheader("Topic Analysis")
        
        topic_id = analysis_results['topic']['id']
        topic_words = analysis_results['topic']['words']
        
        if topic_id != -1:  # -1 is outlier topic in BERTopic
            st.markdown(f"**Assigned Topic:** {topic_id}")
            
            # Topic words
            st.markdown("### Top Topic Words")
            
            # Create word cloud from topic words
            if topic_words:
                word_dict = {word: weight for word, weight in topic_words}
                
                wc = WordCloud(background_color="white", max_words=30, width=800, height=400)
                wc.generate_from_frequencies(word_dict)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
                
                # Display word list
                st.markdown("### Topic Keywords")
                word_list = ", ".join([f"{word} ({weight:.3f})" for word, weight in topic_words[:10]])
                st.write(word_list)
        else:
            st.warning("The speech couldn't be assigned to any specific topic (classified as outlier).")
        
        # Text statistics
        st.subheader("Text Statistics")
        
        # Word count
        word_count = len(new_text.split())
        
        # Unique words
        unique_words = len(set(new_text.lower().split()))
        
        # Average word length
        avg_word_length = sum(len(word) for word in new_text.split()) / word_count if word_count > 0 else 0
        
        # Display stats
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1:
            st.metric("Word Count", word_count)
        with stats_col2:
            st.metric("Unique Words", unique_words)
        with stats_col3:
            st.metric("Avg Word Length", f"{avg_word_length:.1f}")

# Main app function
def main():
    """Main function to run the Streamlit app."""
    render_header()
    
    # Load data and models
    data = load_data()
    models = load_models()
    
    if not data or not models:
        st.error("Failed to load data or models. Please make sure the preprocessing script has been run.")
        return
    
    # Render sidebar and get current page and filtered data
    page, filtered_data = render_sidebar(data)
    
    # Render selected page
    if page == "Dashboard Overview":
        render_dashboard_overview(filtered_data)
    elif page == "Speaker Analysis":
        render_speaker_analysis(filtered_data)
    elif page == "Time Trends":
        render_time_trends(filtered_data)
    elif page == "Topic Modeling":
        render_topic_modeling(filtered_data, models)
    elif page == "Sentiment Comparison":
        render_sentiment_comparison(filtered_data)
    elif page == "New Speech Analysis":
        render_new_speech_analysis(models)

if __name__ == "__main__":
    main()
    
    # Sentiment across decades
    st.subheader("Sentiment Trends Across Decades")
    fig = px.line(
        data['decade_sentiment'],
        x='decade',
        y=['vader_pos', 'vader_neu', 'vader_neg'],
        markers=True,
        labels={'value': 'Sentiment Score', 'decade': 'Decade', 'variable': 'Sentiment Type'},
        color_discrete_map={
            'vader_pos': '#4CAF50',
            'vader_neu': '#9E9E9E',
            'vader_neg': '#F44336'
        }
    )
    fig.update_layout(legend_title="Sentiment Type", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent speeches
    st.subheader("Recent Speeches")
    recent = data['processed'].sort_values('date', ascending=False).head(5)
    for i, row in recent.iterrows():
        sentiment = data['sentiment'][data['sentiment']['date'] == row['date']].iloc[0]['vader_sentiment']
        sentiment_class = f"sentiment-{sentiment}"
        
        expander = st.expander(f"{row['speaker']} ({row['date'].strftime('%Y-%m-%d')})")
        with expander:
            st.markdown(f"**Sentiment:** <span class='{sentiment_class}'>{sentiment.capitalize()}</span>", 
                        unsafe_allow_html=True)
            st.write(row['text'][:300] + "..." if len(row['text']) > 300 else row['text'])

def render_speaker_analysis(data):
    """Render speaker analysis page."""
    st.header("Speaker Analysis")
    
    # Speaker count
    st.subheader("Number of Speeches by Speaker")
    speaker_counts = data['processed']['speaker'].value_counts().reset_index()
    speaker_counts.columns = ['speaker', 'count']
    
    # Only show top 15 if there are more than 15 speakers
    if len(speaker_counts) > 15:
        speaker_counts = speaker_counts.head(15)
    
    fig = px.bar(
        speaker_counts,
        x='count',
        y='speaker',
        orientation='h',
        labels={'count': 'Number of Speeches', 'speaker': 'Speaker'},
        color='count',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Speaker sentiment comparison
    st.subheader("Speaker Sentiment Comparison")
    
    # Filter speaker sentiment for selected speakers or top speakers
    speaker_sentiment = data['speaker_sentiment']
    if len(speaker_sentiment) > 15:
        top_speakers = data['processed']['speaker'].value_counts().nlargest(15).index
        speaker_sentiment = speaker_sentiment[speaker_sentiment['speaker'].isin(top_speakers)]
    
    # Sort by compound sentiment
    speaker_sentiment = speaker_sentiment.sort_values('vader_compound')
    
    fig = px.bar(
        speaker_sentiment,
        x='speaker',
        y='vader_compound',
        labels={'vader_compound': 'Compound Sentiment', 'speaker': 'Speaker'},
        color='vader_compound',
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Speaker sentiment breakdown
    st.subheader("Speaker Sentiment Breakdown")
    
    selected_speaker = st.selectbox(
        "Select a speaker for detailed analysis",
        options=sorted(data['processed']['speaker'].unique())
    )
    
    if selected_speaker:
        speaker_data = data['sentiment'][data['sentiment']['speaker'] == selected_speaker]
        
        if not speaker_data.empty:
            # Sentiment over time for selected speaker
            fig = px.line(
                speaker_data.sort_values('date'),
                x='date',
                y='vader_compound',
                markers=True,
                labels={'vader_compound': 'Compound Sentiment', 'date': 'Date'},
                title=f"Sentiment Trend for {selected_speaker}"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment components for selected speaker
            comp_fig = px.bar(
                speaker_data.sort_values('date').tail(5),
                x='date',
                y=['vader_pos', 'vader_neu', 'vader_neg'],
                barmode='group',
                labels={'value': 'Score', 'date': 'Date', 'variable': 'Sentiment Component'},
                title=f"Recent Sentiment Components for {selected_speaker}",
                color_discrete_map={
                    'vader_pos': '#4CAF50',
                    'vader_neu': '#9E9E9E', 
                    'vader_neg': '#F44336'
                }
            )
            comp_fig.update_layout(height=400)
            st.plotly_chart(comp_fig, use_container_width=True)
            
            # Speech examples for selected speaker
            st.subheader(f"Sample Speeches by {selected_speaker}")
            speaker_speeches = data['processed'][data['processed']['speaker'] == selected_speaker]
            
            for i, row in speaker_speeches.sort_values('date', ascending=False).head(3).iterrows():
                sentiment = data['sentiment'][data['sentiment']['date'] == row['date']].iloc[0]['vader_sentiment']
                sentiment_class = f"sentiment-{sentiment}"
                
                expander = st.expander(f"{row['date'].strftime('%Y-%m-%d')}")
                with expander:
                    st.markdown(f"**Sentiment:** <span class='{sentiment_class}'>{sentiment.capitalize()}</span>", 
                                unsafe_allow_html=True)
                    st.write(row['text'][:500] + "..." if len(row['text']) > 500 else row['text'])

def render_time_trends(data):
    """Render time trends analysis page."""
    st.header("Time Trends Analysis")
    
    # Sentiment over time
    st.subheader("Sentiment Trends Over Time")
    
    # Group by year and calculate average sentiment
    yearly_sentiment = data['sentiment'].groupby('year')[
        ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']
    ].mean().reset_index()
    
    fig = px.line(
        yearly_sentiment,
        x='year',
        y='vader_compound',
        markers=True,
        labels={'vader_compound': 'Compound Sentiment', 'year': 'Year'},
        title="Average Compound Sentiment by Year"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment components over time
    components_fig = px.line(
        yearly_sentiment,
        x='year',
        y=['vader_pos', 'vader_neu', 'vader_neg'],
        markers=True,
        labels={'value': 'Score', 'year': 'Year', 'variable': 'Sentiment Component'},
        title="Sentiment Components Over Time",
        color_discrete_map={
            'vader_pos': '#4CAF50',
            'vader_neu': '#9E9E9E',
            'vader_neg': '#F44336'
        }
    )
    components_fig.update_layout(height=400)
    st.plotly_chart(components_fig, use_container_width=True)
    
    # Sentiment distribution by decade
    st.subheader("Sentiment Distribution by Decade")
    
    # Create a decade-sentiment count table
    decade_sentiment_counts = data['sentiment'].groupby(['decade', 'vader_sentiment']).size().reset_index(name='count')
    
    # Calculate percentage within each decade
    decade_totals = decade_sentiment_counts.groupby('decade')['count'].transform('sum')
    decade_sentiment_counts['percentage'] = decade_sentiment_counts['count'] / decade_totals * 100
    
    # Create stacked bar chart
    fig = px.bar(
        decade_sentiment_counts,
        x='decade',
        y='percentage',
        color='vader_sentiment',
        labels={'percentage': 'Percentage (%)', 'decade': 'Decade'},
        title="Sentiment Distribution by Decade",
        color_discrete_map={
            'positive': '#4CAF50',
            'neutral': '#9E9E9E',
            'negative': '#F44336'
        }
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Major historical events timeline (optional)
    st.subheader("Major Historical Events Timeline")
    
    # Define some example historical events
    events = [
        {"year": 1914, "event": "World War I Begins", "description": "Start of the First World War"},
        {"year": 1929, "event": "Great Depression", "description": "Stock market crash leading to economic depression"},
        {"year": 1939, "event": "World War II Begins", "description": "Start of the Second World War"},
        {"year": 1945, "event": "World War II Ends", "description": "End of the Second World War"},
        {"year": 1963, "event": "JFK Assassination", "description": "President John F. Kennedy assassinated"},
        {"year": 1969, "event": "Moon Landing", "description": "First humans land on the moon"},
        {"year": 1989, "event": "Berlin Wall Falls", "description": "Fall of the Berlin Wall"},
        {"year": 2001, "event": "9/11 Attacks", "description": "Terrorist attacks in the United States"},
        {"year": 2008, "event": "Financial Crisis", "description": "Global financial crisis"},
        {"year": 2020, "event": "COVID-19 Pandemic", "description": "Global pandemic begins"}
    ]
    
    # Filter events based on the year range in the data
    min_year = yearly_sentiment['year'].min()
    max_year = yearly_sentiment['year'].max()
    events = [e for e in events if min_year <= e['year'] <= max_year]
    
    # Create events DataFrame
    events_df = pd.DataFrame(events)
    
    # Create timeline visualization
    fig = px.scatter(
        events_df,
        x='year',
        y=[0] * len(events_df),
        text='event',
        labels={'year': 'Year'},
        title="Timeline of Major Historical Events",
        height=300
    )
    
    fig.update_traces(marker=dict(size=10, color='rgba(0, 0, 255, 0.8)'))
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display events details
    for event in events:
        st.markdown(f"**{event['year']} - {event['event']}:** {event['description']}")

def render_topic_modeling(data, models):
    """Render topic modeling analysis page."""
    st.header("Topic Modeling Analysis")
    
    # Topic distribution
    st.subheader("Topic Distribution")
    
    # Count speeches per topic
    topic_counts = data['topic']['topic'].value_counts().reset_index()
    topic_counts.columns = ['topic', 'count']
    
    # Merge with topic info to get topic names
    topic_info = data['topic_info'].copy()
    topic_info['Name'] = topic_info['Name'].apply(lambda x: x if isinstance(x, str) else "Topic " + str(abs(x)))
    
    # Only use top 10 topics by count and merge with info
    top_topics = topic_counts.head(10)
    top_topics = top_topics.merge(topic_info[['Topic', 'Name']], left_on='topic', right_on='Topic', how='left')
    
    fig = px.bar(
        top_topics,
        x='count',
        y='Name',
        orientation='h',
        labels={'count': 'Number of Speeches', 'Name': 'Topic'},
        title="Top 10 Topics",
        color='count',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Topic words visualization
    st.subheader("Topic Words")
    
    # Let user select a topic to examine
    topic_options = topic_info.sort_values('Topic')['Name'].tolist()
    selected_topic_name = st.selectbox("Select a topic to examine", options=topic_options)
    
    selected_topic_id = topic_info[topic_info['Name'] == selected_topic_name]['Topic'].iloc[0]
    
    # Get topic words
    topic_words = models['topic_model'].get_topic(selected_topic_id)
    
    # Create word cloud
    if topic_words:
        word_dict = {word: weight for word, weight in topic_words}
        
        # Create and display word cloud
        wc = WordCloud(background_color="white", max_words=50, width=800, height=400)
        wc.generate_from_frequencies(word_dict)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
    
    # Topic evolution over time
    st.subheader("Topic Evolution Over Time")
    
    # Group by year and count topics
    topic_by_year = data['topic'].groupby(['year', 'topic']).size().reset_index(name='count')
    
    # Merge with topic info
    topic_by_year = topic_by_year.merge(
        topic_info[['Topic', 'Name']], 
        left_on='topic', 
        right_on='Topic', 
        how='left'
    )
    
    # Select top topics for visualization
    top_topic_ids = topic_counts.head(5)['topic'].tolist()
    top_topics_by_year = topic_by_year[topic_by_year['topic'].isin(top_topic_ids)]
    
    # Create line chart
    fig = px.line(
        top_topics_by_year,
        x='year',
        y='count',
        color='Name',
        markers=True,
        labels={'count': 'Number of Speeches', 'year': 'Year', 'Name': 'Topic'},
        title="Top Topics Evolution Over Time"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Topic-Sentiment Relationship
    st.subheader("Topic-Sentiment Relationship")
    
    # Merge topic and sentiment data
    topic_sentiment = pd.merge(
        data['topic'][['speaker', 'date', 'topic']],
        data['sentiment'][['speaker', 'date', 'vader_compound', 'vader_sentiment']],
        on=['speaker', 'date']
    )
    
    # Group by topic and calculate average sentiment
    topic_sentiment_avg = topic_sentiment.groupby('topic')['vader_compound'].mean().reset_index()
    
    # Merge with topic info
    topic_sentiment_avg = topic_sentiment_avg.merge(
        topic_info[['Topic', 'Name']], 
        left_on='topic', 
        right_on='Topic', 
        how='left'
    )
    
    # Sort by sentiment
    topic_sentiment_avg = topic_sentiment_avg.sort_values('vader_compound')
    
    # Create bar chart
    fig = px.bar(
        topic_sentiment_avg.head(10),
        x='Name',
        y='vader_compound',
        labels={'vader_compound': 'Average Compound Sentiment', 'Name': 'Topic'},
        title="Average Sentiment by Topic",
        color='vader_compound',
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)