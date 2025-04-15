import os
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from textblob import TextBlob
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import streamlit as st

# --- Load Environment Variables ---
load_dotenv()

# --- Set up Logging ---
logging.basicConfig(level=logging.DEBUG)

# --- Setup Google Gemini API ---
def setup_gemini_api():
    """
    Set up the Gemini API with the API key from environment variables.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    return genai.Client(api_key=api_key)

# --- Fetch News Headlines ---
def fetch_news_headlines(ticker, max_headlines=10):
    """
    Fetch recent news headlines for a given stock ticker.
    """
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key:
        raise ValueError("NEWS_API_KEY not found in environment variables")
    
    today = datetime.now()
    one_month_ago = today - timedelta(days=30)
    from_date = one_month_ago.strftime('%Y-%m-%d')
    to_date = today.strftime('%Y-%m-%d')
    
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': ticker,
        'apiKey': api_key,
        'language': 'en',
        'sortBy': 'publishedAt',
        'from': from_date,
        'to': to_date,
        'pageSize': max_headlines
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == 'ok' and data['totalResults'] > 0:
            return data['articles'][:max_headlines]
        else:
            logging.warning(f"No news found for ticker {ticker}.")
            return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching news headlines: {str(e)}")
        raise Exception(f"Error fetching news headlines: {str(e)}")

# --- Sentiment Analysis ---
def analyze_sentiment(headline):
    """
    Analyze the sentiment of a news headline using Google's Gemini API.
    """
    try:
        client = setup_gemini_api()
        model_name = "gemini-2.0-flash"
        prompt = f"""
        You are a financial sentiment analyzer. Evaluate the sentiment of this headline about a stock:
        
        "{headline}"
        
        Return ONLY a number between -10 and +10:
        """
        
        response = client.models.generate_content(model=model_name, contents=prompt)
        sentiment_text = response.text.strip()
        sentiment_text = ''.join(c for c in sentiment_text if c.isdigit() or c in ['-', '.'])
        
        try:
            sentiment_score = float(sentiment_text)
            sentiment_score = max(-10, min(10, sentiment_score))
            return sentiment_score
        except ValueError:
            return 1
    except Exception as e:
        print(f"Error analyzing sentiment: {str(e)}")
        return 1

# --- Replace Zero Sentiment Scores ---
def replace_zero_score(score, default_positive=1, default_negative=-1):
    """
    Replace a zero sentiment score with a default positive or negative value.
    """
    if score == 0:
        return default_positive
    elif score > -0.1 and score < 0.1:
        if score < 0:
            return default_negative
        else:
            return default_positive
    return score

# --- Streamlit Application ---
# Set page configuration
st.set_page_config(
    page_title="Stock News Sentiment Analysis",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("Stock News Sentiment Analysis")
st.markdown("""
This application analyzes sentiment for the latest news headlines related to a stock ticker.
Enter a stock ticker symbol (e.g., AAPL, TSLA, GOOGL) to get started.
""")

# Input for stock ticker
ticker_input = st.text_input("Enter Stock Ticker Symbol:", placeholder="e.g., AAPL")

# When a ticker is submitted
if ticker_input:
    ticker = ticker_input.strip().upper()
    
    with st.spinner(f"Fetching news headlines for {ticker}..."):
        try:
            # Fetch news headlines
            headlines = fetch_news_headlines(ticker)
            
            if not headlines:
                st.error(f"No news headlines found for ticker {ticker}. Please check if the ticker is valid.")
            else:
                # Create a dataframe to store headlines and sentiment scores
                df = pd.DataFrame({
                    'headline': [headline['title'] for headline in headlines],
                    'url': [headline['url'] for headline in headlines],
                    'source': [headline['source']['name'] for headline in headlines],
                    'publishedAt': [headline['publishedAt'] for headline in headlines]
                })
                
                # Analyze sentiment for each headline
                st.subheader(f"Analyzing sentiment for {ticker} headlines...")
                progress_bar = st.progress(0)
                
                sentiment_scores = []
                for i, headline in enumerate(df['headline']):
                    with st.spinner(f"Analyzing headline {i+1}/{len(df)}..."):
                        score = analyze_sentiment(headline)
                        score = replace_zero_score(score)
                        sentiment_scores.append(score)
                    progress_bar.progress((i + 1) / len(df))
                
                df['sentiment_score'] = sentiment_scores
                
                # Calculate average sentiment score
                avg_sentiment = df['sentiment_score'].mean()
                
                # Display results
                st.success("Analysis completed!")
                
                # Display average sentiment with colorful indicator
                st.subheader("Overall Sentiment Analysis")
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if avg_sentiment > 0:
                        st.markdown(f"### 📈 Positive")
                        sentiment_color = "green"
                    else:
                        st.markdown(f"### 📉 Negative")
                        sentiment_color = "red"
                    
                    st.metric(
                        label="Average Sentiment Score", 
                        value=f"{avg_sentiment:.2f}",
                        delta=f"{avg_sentiment:.2f}"
                    )
                
                with col2:
                    # Create a gauge chart for average sentiment
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=avg_sentiment,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"Sentiment Score for {ticker}"},
                        gauge={
                            'axis': {'range': [-10, 10]},
                            'bar': {'color': sentiment_color},
                            'steps': [
                                {'range': [-10, -5], 'color': 'rgba(255, 0, 0, 0.3)'},
                                {'range': [-5, 0], 'color': 'rgba(255, 165, 0, 0.3)'},
                                {'range': [0, 5], 'color': 'rgba(144, 238, 144, 0.3)'},
                                {'range': [5, 10], 'color': 'rgba(0, 128, 0, 0.3)'}
                            ],
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display headlines with their sentiment scores
                st.subheader("Analyzed Headlines")
                
                # Format the dataframe for display
                display_df = df.copy()
                display_df['sentiment_category'] = display_df['sentiment_score'].apply(
                    lambda x: "Positive" if x > 0 else "Negative"
                )
                display_df['headline_with_link'] = display_df.apply(
                    lambda row: f"<a href='{row['url']}' target='_blank'>{row['headline']}</a>", 
                    axis=1
                )
                
                # Display the headlines in an expander
                with st.expander("View All Headlines", expanded=True):
                    for i, row in display_df.iterrows():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(row['headline_with_link'], unsafe_allow_html=True)
                            st.caption(f"Source: {row['source']} | Published: {row['publishedAt']}")
                        with col2:
                            if row['sentiment_score'] > 0:
                                st.markdown(f"#### 😀 {row['sentiment_score']:.1f}")
                            else:
                                st.markdown(f"#### 😞 {row['sentiment_score']:.1f}")
                        st.divider()
                
                # Visualizations
                st.subheader("Sentiment Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart for sentiment scores by headline
                    display_df['headline_short'] = display_df['headline'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
                    bar_fig = px.bar(
                        display_df,
                        x='headline_short',
                        y='sentiment_score',
                        color='sentiment_score',
                        color_continuous_scale='RdYlGn',
                        labels={'headline_short': 'Headline', 'sentiment_score': 'Sentiment Score'},
                        title=f'Sentiment Scores for {ticker} Headlines'
                    )
                    bar_fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(bar_fig, use_container_width=True)
                
                with col2:
                    # Pie chart showing distribution of positive vs negative sentiment
                    sentiment_counts = display_df['sentiment_category'].value_counts().reset_index()
                    sentiment_counts.columns = ['Sentiment', 'Count']
                    
                    pie_fig = px.pie(
                        sentiment_counts, 
                        values='Count', 
                        names='Sentiment',
                        color='Sentiment',
                        color_discrete_map={'Positive': 'green', 'Negative': 'red'},
                        title=f'Sentiment Distribution for {ticker}'
                    )
                    st.plotly_chart(pie_fig, use_container_width=True)
                
                # Sentiment distribution histogram
                hist_fig = px.histogram(
                    display_df,
                    x='sentiment_score',
                    nbins=20,
                    color_discrete_sequence=['lightblue'],
                    labels={'sentiment_score': 'Sentiment Score'},
                    title=f'Distribution of Sentiment Scores for {ticker}'
                )
                hist_fig.add_vline(x=avg_sentiment, line_dash="dash", line_color="red", annotation_text=f"Avg: {avg_sentiment:.2f}")
                hist_fig.update_layout(bargap=0.1)
                st.plotly_chart(hist_fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

# Footer
st.markdown("---")
st.caption("Stock News Sentiment Analysis Application | Data is fetched in real-time")
