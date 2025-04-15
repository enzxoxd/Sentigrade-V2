import os
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
import logging
import streamlit as st
import time
from typing import Optional

# --- Load Environment Variables ---
load_dotenv()

# --- Set up Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Setup Google Gemini API ---
def setup_gemini_api():
    """
    Set up the Gemini API with the API key from environment variables.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")
        return None
    return api_key

# --- Sentiment Analysis ---
def gemini_analyze_sentiment(text: str, api_key: Optional[str]) -> float:
    """
    Analyze sentiment of text using Google Gemini API.
    
    Args:
        text (str): Text to analyze sentiment for
        api_key (Optional[str]): Gemini API Key
        
    Returns:
        float: Sentiment score between -10 (negative) and 10 (positive)
    """
    if not api_key:
        st.error("Gemini API Key not configured.")
        return 0.0
    
    try:
        # Initialize the Gemini client with the provided API key
        genai.configure(api_key=api_key)
        
        # Set up the model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Request Gemini to return only a sentiment score
        prompt = f"""Please analyze the sentiment of the following headline and return ONLY integer 
        between -10 (very negative) and 10 (very positive). The number should reflect the sentiment score.
        No explanation, just a number.
        
        Headline: {text}
        """
        
        response = model.generate_content(prompt)
        
        # Extract the sentiment score and convert to float
        try:
            sentiment_score = float(response.text.strip())
            
            # Ensure the score is in the range [-10, 10]
            sentiment_score = max(-10, min(10, sentiment_score))
            
            # Normalize to range [-1, 1] for consistency with our app
            normalized_score = sentiment_score 
            
            return normalized_score
            
        except ValueError:
            st.warning(f"Could not convert sentiment response to number: {response.text}")
            return 0.0
            
    except Exception as e:
        st.error(f"Error with Gemini API: {str(e)}")
        return 0.0

# --- Calculate Average Sentiment (Handling None values) ---
def calculate_average_sentiment(scores):
    """
    Calculate average sentiment score, handling None values.
    
    Args:
        scores (list): List of sentiment scores that may include None values
        
    Returns:
        float: Average sentiment score
    """
    valid_scores = [score for score in scores if score is not None]
    if not valid_scores:
        return 0
    return sum(valid_scores) / len(valid_scores)

# --- Fetch News Headlines ---
def fetch_news_headlines(ticker, from_date=None, to_date=None):
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        st.error("NEWSAPI_KEY not found. Please add it to your .env file.")
        return []

    # Define only business-related sources
    business_sources = "bloomberg,the-wall-street-journal,forbes,fortune,axios,financial-times,cnbc,business-insider"

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&"
        f"sources={business_sources}&"
        f"apiKey={api_key}&"
        f"pageSize=10&"
        f"sortBy=publishedAt&"
        f"language=en"
    )


    # Add date filters to the URL if specified
    if from_date:
        url += f"&from={from_date}"
    if to_date:
        url += f"&to={to_date}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()
        articles = news_data.get('articles', [])
        if not articles:
            st.warning(f"No news found for ticker: {ticker}. Try another one.")
        return articles
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {str(e)}")
        return []

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
# Date Range Filter
st.markdown("### Filter by Date Range")
today = datetime.today()
default_start = today - timedelta(days=7)
start_date = st.date_input("Start Date", default_start)
end_date = st.date_input("End Date", today)

# When a ticker is submitted
if ticker_input:
    ticker = ticker_input.strip().upper()

    with st.spinner(f"Fetching news headlines for {ticker}..."):
        try:
            # Fetch news headlines using the newly defined function
            headlines = fetch_news_headlines(ticker, from_date=start_date.strftime("%Y-%m-%d"), to_date=end_date.strftime("%Y-%m-%d"))

            
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
                
                # Fetch the Gemini API key
                api_key = setup_gemini_api()

                sentiment_scores = []
                for i, headline in enumerate(df['headline']):
                    with st.spinner(f"Analyzing headline {i+1}/{len(df)}..."):
                        score = gemini_analyze_sentiment(headline, api_key)
                        sentiment_scores.append(score)
                    progress_bar.progress((i + 1) / len(df))
                
                df['sentiment_score'] = sentiment_scores
                
                # Calculate average sentiment score (only from valid scores)
                avg_sentiment = calculate_average_sentiment(df['sentiment_score'])
                
                # Count headlines with valid sentiment scores
                valid_scores_count = sum(1 for score in sentiment_scores if score is not None)
                total_headlines = len(sentiment_scores)
                
                # Display results
                st.success(f"Analysis completed! Analyzed {valid_scores_count} out of {total_headlines} headlines.")
                
                if valid_scores_count == 0:
                    st.error("Could not analyze any headlines. Please try again later.")
                else:
                    # Display average sentiment with colorful indicator
                    st.subheader("Overall Sentiment Analysis")
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if avg_sentiment > 0:
                            st.markdown(f"### 📈 Positive")
                            sentiment_color = "green"
                        elif avg_sentiment < 0:
                            st.markdown(f"### 📉 Negative")
                            sentiment_color = "red"
                        else:
                            st.markdown(f"### ⚖️ Neutral")
                            sentiment_color = "gray"
                        
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
                    
                    # Filter out rows with None sentiment scores
                    display_df = df.dropna(subset=['sentiment_score']).copy()
                    
                    if display_df.empty:
                        st.warning("No headlines with valid sentiment scores to display.")
                    else:
                        # Format the dataframe for display
                        display_df['sentiment_category'] = display_df['sentiment_score'].apply(
                            lambda x: "Positive" if x > 0 else ("Neutral" if x == 0 else "Negative")
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
                                    if row['sentiment_score'] > 3:
                                        st.markdown(f"#### 😀 {row['sentiment_score']:.1f}")
                                    elif row['sentiment_score'] > 0:
                                        st.markdown(f"#### 🙂 {row['sentiment_score']:.1f}")
                                    elif row['sentiment_score'] == 0:
                                        st.markdown(f"#### 😐 {row['sentiment_score']:.1f}")
                                    elif row['sentiment_score'] > -3:
                                        st.markdown(f"#### 🙁 {row['sentiment_score']:.1f}")
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
                                color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'},
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
