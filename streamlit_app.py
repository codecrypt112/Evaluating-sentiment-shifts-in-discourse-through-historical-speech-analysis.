import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import plotly.express as px
from bertopic import BERTopic

st.set_page_config(layout="wide")
st.title("📜 Historical Speech Analysis Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("historical_speeches.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    return df

df = load_data()

st.sidebar.header("Filters")
year_range = st.sidebar.slider("Select Year Range", int(df['year'].min()), int(df['year'].max()), (int(df['year'].min()), int(df['year'].max())))
df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

st.subheader("🕰 Distribution of Speeches Over Time")
fig1, ax = plt.subplots(figsize=(12, 4))
sns.countplot(data=df_filtered, x='year', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig1)

st.subheader("☁️ Word Cloud of Speeches")
text = " ".join(df_filtered['text'].tolist())
wordcloud = WordCloud(width=800, height=300, background_color='white').generate(text)
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.imshow(wordcloud, interpolation='bilinear')
ax2.axis("off")
st.pyplot(fig2)

st.subheader("📈 Sentiment Trend Over Time")
analyzer = SentimentIntensityAnalyzer()
df_filtered['sentiment'] = df_filtered['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
fig3 = px.line(df_filtered, x='date', y='sentiment', title='Sentiment Over Time')
st.plotly_chart(fig3, use_container_width=True)

st.subheader("🧠 Topic Modeling with BERTopic")
if st.button("Generate Topics"):
    with st.spinner("Training BERTopic model..."):
        topic_model = BERTopic()
        topics, _ = topic_model.fit_transform(df_filtered['text'].tolist())
        st.success("Topics Generated!")
        st.plotly_chart(topic_model.visualize_barchart(top_n_topics=5), use_container_width=True)

st.markdown("---")
st.caption("Created by ChatGPT for Historical Speech Analysis.")
