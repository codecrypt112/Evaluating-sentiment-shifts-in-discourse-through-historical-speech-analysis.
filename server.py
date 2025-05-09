#!/usr/bin/env python3
"""
server.py - Streamlit Dashboard for Historical Speech Sentiment Analysis

This script creates an interactive Streamlit dashboard to visualize the results
of sentiment analysis on historical speeches.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import json
import pickle
from datetime import datetime
import os
from bertopic import BERTopic
import gensim
from gensim import corpora
from tensorflow.keras.models import load_model
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import nltk
from PIL import Image
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Historical Speech Sentiment Analysis",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    .highlight {
        background-color: #DBEAFE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .footnote {
        font-size: 0.8rem;
        color: #6B7280;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Path constants
DATA_DIR = "data"
MODELS_DIR = "models"
VISUALIZATIONS_DIR = "visualizations"

# Check if required directories exist
if not os.path.exists(DATA_DIR):
    st.error("Data directory not found. Please run train.py first.")
    st.stop()

# Load data
@st.cache_data
def load_data():
    """Load all processed data files"""
    data = {
        'speeches': pd.read_csv(f"{DATA_DIR}/speeches_with_sentiment.csv"),
        'time_sentiment': pd.read_csv(f"{DATA_DIR}/time_sentiment.csv"),
        'speaker_sentiment': pd.read_csv(f"{DATA_DIR}/speaker_sentiment.csv"),
        'decade_sentiment': pd.read_csv(f"{DATA_DIR}/decade_sentiment.csv"),
        'topic_time': pd.read_csv(f"{DATA_DIR}/topic_time.csv"),
        'entity_counts': pd.read_csv(f"{DATA_DIR}/entity_counts.csv"),
        'decade_texts': pd.read_csv(f"{DATA_DIR}/decade_texts.csv"),
        'significant_shifts': pd.read_csv(f"{DATA_DIR}/significant_shifts.csv"),
        'sentiment_volatility': pd.read_csv(f"{DATA_DIR}/sentiment_volatility.csv"),
        'named_entities': pd.read_csv(f"{DATA_DIR}/named_entities.csv")
    }
    
    # Convert date columns to datetime
    for df in ['speeches', 'significant_shifts']:
        if 'date' in data[df].columns:
            data[df]['date'] = pd.to_datetime(data[df]['date'])
    
    # Load metadata
    with open(f"{DATA_DIR}/metadata.json", 'r') as f:
        data['metadata'] = json.load(f)
    
    return data

# Create word cloud
@st.cache_data
def generate_wordcloud(text):
    """Generate a word cloud from the text"""
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100,
        contour_width=3,
        contour_color='steelblue'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

# Generate time series plot
def plot_sentiment_timeline(df, model_type='vader'):
    """Generate a time series plot of sentiment"""
    fig = px.line(
        df, 
        x='year', 
        y=[f'{model_type}_compound', f'{model_type}_positive', f'{model_type}_negative'],
        title=f"Sentiment Over Time ({model_type.upper()})",
        labels={'value': 'Sentiment Score', 'year': 'Year'},
        color_discrete_map={
            f'{model_type}_compound': '#3B82F6', 
            f'{model_type}_positive': '#10B981', 
            f'{model_type}_negative': '#EF4444'
        }
    )
    
    fig.update_layout(
        legend_title_text='Sentiment Type',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    return fig

# Generate speaker comparison
def plot_speaker_comparison(df, top_n=15):
    """Generate a comparison of sentiment across top speakers"""
    top_speakers = df.sort_values('vader_compound', ascending=False).head(top_n)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_speakers['speaker'],
        y=top_speakers['vader_compound'],
        name='VADER',
        marker_color='#3B82F6',
        opacity=0.8
    ))
    
    fig.add_trace(go.Bar(
        x=top_speakers['speaker'],
        y=top_speakers['bert_compound'],
        name='BERT',
        marker_color='#8B5CF6',
        opacity=0.8
    ))
    
    fig.update_layout(
        title="Top Speakers by Sentiment Score",
        xaxis_title="Speaker",
        yaxis_title="Compound Sentiment Score",
        barmode='group',
        xaxis={'categoryorder':'total descending'},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600
    )
    
    return fig

# Generate decade comparison
def plot_decade_sentiment(df):
    """Plot average sentiment by decade."""
    fig = px.bar(
        df,
        x='decade',
        y=['vader_compound', 'bert_compound'],
        barmode='group',
        title="Average Sentiment by Decade",
        labels={'value': 'Sentiment Score', 'decade': 'Decade', 'variable': 'Model'},
        color_discrete_map={
            'vader_compound': '#3B82F6',
            'bert_compound': '#10B981'
        }
    )
    fig.update_layout(
        legend_title_text='Model',
        hovermode="x unified"
    )
    return fig

# Generate significant shifts plot
def plot_significant_shifts(df):
    """Generate a visualization of significant shifts in sentiment"""
    fig = px.scatter(
        df,
        x='date',
        y=['vader_shift', 'bert_shift'],
        color='speaker',
        size=df['vader_shift'] + df['bert_shift'],
        hover_name='speaker',
        title="Significant Shifts in Sentiment",
        labels={
            'date': 'Date',
            'value': 'Magnitude of Shift',
            'variable': 'Model'
        },
        height=600
    )
    
    fig.update_layout(
        legend_title_text='Speaker',
        hovermode="closest"
    )
    
    return fig

# Generate topic distribution over time
def plot_topic_distribution(df, topic_model):
    """Generate a visualization of topic distribution over time"""
    # Prepare data
    pivot_df = df.pivot_table(
        index='decade', 
        columns='topic', 
        values='count', 
        aggfunc='sum',
        fill_value=0
    )
    
    # Normalize by decade
    for decade in pivot_df.index:
        pivot_df.loc[decade] = pivot_df.loc[decade] / pivot_df.loc[decade].sum()
    
    # Get topic labels if available
    try:
        topic_labels = [f"Topic {i}: {', '.join(words[:3])}" 
                        for i, (_, words, _) in enumerate(topic_model.get_topics())]
    except:
        topic_labels = [f"Topic {i}" for i in pivot_df.columns]
    
    # Create figure
    fig = px.area(
        pivot_df.reset_index().melt(id_vars='decade'),
        x='decade', 
        y='value',
        color='topic',
        title="Topic Distribution Over Time",
        labels={'value': 'Proportion', 'decade': 'Decade', 'topic': 'Topic'}
    )
    
    fig.update_layout(
        legend_title_text='Topic',
        hovermode="x unified",
        height=600
    )
    
    return fig

# Generate entity type distribution
def plot_entity_distribution(df):
    """Generate a visualization of named entity types"""
    # Get top 10 entity types
    top_entities = df.sort_values('count', ascending=False).head(10)
    
    fig = px.bar(
        top_entities,
        x='entity_type',
        y='count',
        title="Top Named Entity Types",
        labels={'entity_type': 'Entity Type', 'count': 'Frequency'},
        color='count',
        color_continuous_scale=px.colors.sequential.Blues
    )
    
    fig.update_layout(
        xaxis={'categoryorder':'total descending'},
        height=500
    )
    
    return fig

# Generate sentiment volatility plot
def plot_sentiment_volatility(df):
    """Generate a visualization of sentiment volatility over time"""
    fig = px.bar(
        df,
        x='decade',
        y=['vader_volatility', 'bert_volatility'],
        title="Sentiment Volatility by Decade",
        labels={
            'decade': 'Decade', 
            'value': 'Volatility (Standard Deviation)',
            'variable': 'Model'
        },
        barmode='group',
        color_discrete_map={
            'vader_volatility': '#3B82F6', 
            'bert_volatility': '#8B5CF6'
        }
    )
    
    fig.update_layout(
        legend_title_text='Model',
        hovermode="x unified",
        height=500
    )
    
    return fig

# Main application
def main():
    """Main Streamlit application"""
    # Load data
    try:
        data = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error("Please make sure you've run train.py first.")
        st.stop()
    
    # Header
    st.markdown('<h1 class="main-header">Historical Speech Sentiment Analysis</h1>', unsafe_allow_html=True)
    
    # Dashboard introduction
    with st.expander("About this Dashboard", expanded=False):
        st.markdown("""
        <div class="highlight">
        This dashboard presents an analysis of sentiment trends in historical speeches using multiple machine learning approaches:
        
        - **VADER** (Valence Aware Dictionary and sEntiment Reasoner) - A lexicon and rule-based sentiment analysis tool
        - **BERT** (Bidirectional Encoder Representations from Transformers) - A transformer-based NLP model
        - **LSTM** (Long Short-Term Memory) - A recurrent neural network architecture
        - **BERTopic** - A topic modeling technique leveraging BERT embeddings
        
        The analysis covers speeches from various periods and speakers, exploring how sentiment and topics have evolved over time.
        </div>
        """, unsafe_allow_html=True)
    
    # Display metadata
    metadata = data['metadata']
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Number of Speeches", metadata['num_speeches'])
    col2.metric("Unique Speakers", metadata['num_speakers'])
    col3.metric("Date Range", f"{metadata['date_range'][0]} to {metadata['date_range'][1]}")
    col4.metric("Named Entities", metadata['num_entities'])
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Sentiment Over Time", 
        "Speaker Analysis", 
        "Topic Modeling", 
        "Named Entities",
        "Advanced Analysis"
    ])
    
    # Tab 1: Sentiment Over Time
    with tab1:
        st.markdown('<h2 class="section-header">Sentiment Trends Through History</h2>', unsafe_allow_html=True)
        
        # Time series plot
        st.markdown('<h3 class="subsection-header">Sentiment Timeline</h3>', unsafe_allow_html=True)
        model_type = st.radio(
            "Select sentiment model:",
            ["vader", "bert"],
            format_func=lambda x: "VADER" if x == "vader" else "BERT",
            horizontal=True
        )
        st.plotly_chart(plot_sentiment_timeline(data['time_sentiment'], model_type), use_container_width=True)
        
        # Decade comparison
        st.markdown('<h3 class="subsection-header">Sentiment by Decade</h3>', unsafe_allow_html=True)
        st.plotly_chart(plot_decade_sentiment(data['decade_sentiment']), use_container_width=True)
        
        # Word clouds by decade
        st.markdown('<h3 class="subsection-header">Word Clouds by Decade</h3>', unsafe_allow_html=True)
        selected_decade = st.selectbox(
            "Select decade:",
            sorted(data['decade_texts']['decade'].unique())
        )
        
        decade_text = data['decade_texts'][data['decade_texts']['decade'] == selected_decade]['text'].values[0]
        st.pyplot(generate_wordcloud(decade_text))
    
    # Tab 2: Speaker Analysis
    with tab2:
        st.markdown('<h2 class="section-header">Speaker Sentiment Analysis</h2>', unsafe_allow_html=True)
        
        # Speaker comparison
        st.markdown('<h3 class="subsection-header">Top Speakers by Sentiment</h3>', unsafe_allow_html=True)
        top_n = st.slider("Number of speakers to display:", 5, 20, 10)
        st.plotly_chart(plot_speaker_comparison(data['speaker_sentiment'], top_n), use_container_width=True)
        
        # Individual speaker analysis
        st.markdown('<h3 class="subsection-header">Individual Speaker Analysis</h3>', unsafe_allow_html=True)
        selected_speaker = st.selectbox(
            "Select speaker:",
            sorted(data['speeches']['speaker'].unique())
        )
        
        speaker_speeches = data['speeches'][data['speeches']['speaker'] == selected_speaker]
        
        # Display speaker stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Speeches", len(speaker_speeches))
        col2.metric("Average VADER Sentiment", f"{speaker_speeches['vader_compound'].mean():.3f}")
        col3.metric("Average BERT Sentiment", f"{speaker_speeches['bert_compound'].mean():.3f}")
        
        # Display speeches
        st.markdown('<h4>Speeches</h4>', unsafe_allow_html=True)
        for _, speech in speaker_speeches.iterrows():
            with st.expander(f"{speech['date'].strftime('%Y-%m-%d')}"):
                st.write(speech['text'])
                st.markdown(f"""
                **VADER Sentiment:** {speech['vader_compound']:.3f} 
                (Positive: {speech['vader_positive']:.3f}, 
                Negative: {speech['vader_negative']:.3f}, 
                Neutral: {speech['vader_neutral']:.3f})
                
                **BERT Sentiment:** {speech['bert_compound']:.3f}
                (Positive: {speech['bert_positive']:.3f}, 
                Negative: {speech['bert_negative']:.3f})
                
                **Topic:** {speech['topic']}
                """)
    
    # Tab 3: Topic Modeling
    with tab3:
        st.markdown('<h2 class="section-header">Topic Analysis</h2>', unsafe_allow_html=True)
        
        # Load topic model if available
        topic_model = None
        try:
            topic_model = BERTopic.load("models/bertopic_model")
            st.success("BERTopic model loaded successfully")
        except:
            st.warning("BERTopic model not found. Using only numeric topic IDs.")
        
        # Topic distribution over time
        st.markdown('<h3 class="subsection-header">Topic Distribution Over Time</h3>', unsafe_allow_html=True)
        st.plotly_chart(plot_topic_distribution(data['topic_time'], topic_model), use_container_width=True)
        
        # Topic details
        st.markdown('<h3 class="subsection-header">Topic Details</h3>', unsafe_allow_html=True)
        
        if topic_model:
            # Display topic info from the model
            topic_info = topic_model.get_topic_info()
            
            for topic_id in sorted(data['speeches']['topic'].unique()):
                if topic_id != -1:  # Skip outlier topic
                    with st.expander(f"Topic {topic_id}"):
                        # Get top words for this topic
                        topic_words = topic_model.get_topic(topic_id)
                        st.markdown("#### Top Words")
                        words_df = pd.DataFrame(topic_words, columns=["Word", "Score"])
                        st.dataframe(words_df.head(10))
                        
                        # Get representative documents
                        topic_speeches = data['speeches'][data['speeches']['topic'] == topic_id]
                        st.markdown(f"#### Representative Speeches ({len(topic_speeches)} speeches)")
                        
                        if not topic_speeches.empty:
                            sample_speech = topic_speeches.sample(1).iloc[0]
                            st.markdown(f"**Speaker:** {sample_speech['speaker']}")
                            st.markdown(f"**Date:** {sample_speech['date'].strftime('%Y-%m-%d')}")
                            st.markdown(f"**Text:** {sample_speech['text']}")
        else:
            # Just display topic distribution
            topic_dist = data['speeches']['topic'].value_counts().sort_index()
            st.bar_chart(topic_dist)
    
    # Tab 4: Named Entities
    with tab4:
        st.markdown('<h2 class="section-header">Named Entity Analysis</h2>', unsafe_allow_html=True)
        
        # Entity type distribution
        st.markdown('<h3 class="subsection-header">Entity Type Distribution</h3>', unsafe_allow_html=True)
        st.plotly_chart(plot_entity_distribution(data['entity_counts']), use_container_width=True)
        
        # Entity exploration
        st.markdown('<h3 class="subsection-header">Explore Entities</h3>', unsafe_allow_html=True)
        
        entity_type = st.selectbox(
            "Select entity type:",
            sorted(data['named_entities']['type'].unique())
        )
        
        filtered_entities = data['named_entities'][data['named_entities']['type'] == entity_type]
        entity_counts = filtered_entities['text'].value_counts().reset_index()
        entity_counts.columns = ['Entity', 'Count']
        
        st.dataframe(
            entity_counts.head(20),
            column_config={
                "Entity": st.column_config.TextColumn("Entity"),
                "Count": st.column_config.NumberColumn("Occurrences")
            },
            hide_index=True
        )
    
    # Tab 5: Advanced Analysis
    with tab5:
        st.markdown('<h2 class="section-header">Advanced Analysis</h2>', unsafe_allow_html=True)
        
        # Sentiment volatility
        st.markdown('<h3 class="subsection-header">Sentiment Volatility</h3>', unsafe_allow_html=True)
        st.markdown("""
        This chart shows how much sentiment fluctuated within each decade, measured by the 
        standard deviation of sentiment scores.
        """)
        st.plotly_chart(plot_sentiment_volatility(data['sentiment_volatility']), use_container_width=True)
        
        # Significant shifts
        st.markdown('<h3 class="subsection-header">Significant Sentiment Shifts</h3>', unsafe_allow_html=True)
        st.markdown("""
        This visualization highlights moments in history when there were significant shifts in sentiment
        between consecutive speeches.
        """)
        st.plotly_chart(plot_significant_shifts(data['significant_shifts']), use_container_width=True)
        
        # Correlation analysis
        st.markdown('<h3 class="subsection-header">Correlation Analysis</h3>', unsafe_allow_html=True)
        
        numeric_cols = [
            'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral',
            'bert_compound', 'bert_positive', 'bert_negative',
            'year', 'decade', 'century', 'topic'
        ]
        
        corr_matrix = data['speeches'][numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        plt.title('Correlation Matrix of Sentiment Features')
        st.pyplot(fig)
    
    # Footer
    st.markdown("""
    <div class="footnote">
    Analysis performed using VADER, BERT, LSTM, and BERTopic models. 
    Dashboard created with Streamlit.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()