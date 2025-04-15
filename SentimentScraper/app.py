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
    return genai.Client(api_key=api_key)

# --- Fetch News Headlines ---
def fetch_news_headlines(ticker, max_headlines=10):
    """
    Fetch recent news headlines for a given stock ticker.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., AAPL)
        max_headlines (int): Maximum number of headlines to fetch
        
    Returns:
        list: List of news articles
    """
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key:
        st.error("NEWS_API_KEY not found in environment variables. Please add it to your .env file.")
        return []
    
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
            logger.warning(f"No news found for ticker {ticker}.")
            return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching news headlines: {str(e)}")
        st.error(f"Error fetching news headlines: {str(e)}")
        return []

# --- Sentiment Analysis ---
def analyze_sentiment(headline, retry_count=3, backoff_factor=2):
    """
    Analyze the sentiment of a news headline using Google's Gemini API.
    Includes retry logic with exponential backoff.
    
    Args:
        headline (str): News headline text
        retry_count (int): Number of retries if API call fails
        backoff_factor (int): Factor for exponential backoff between retries
        
    Returns:
        float: Sentiment score between -10 and +10
    """
    client = setup_gemini_api()
    if not client:
        st.error("Unable to set up Gemini API. Check your API key.")
        return None
    
    model_name = "gemini-2.0-flash"
    prompt = f"""
    You are a financial sentiment analyzer. Evaluate the sentiment of this headline about a stock:
    
    "{headline}"
    
    Return ONLY a number between -10 and +10 where:
    -10 = extremely negative (catastrophic news)
    -5 = moderately negative
    0 = neutral
    +5 = moderately positive
    +10 = extremely positive (breakthrough news)
    
    Provide ONLY the number. No other text.
    """
    
    for attempt in range(retry_count):
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            sentiment_text = response.text.strip()
            
            # Extract numeric value (handling possible text around the number)
            sentiment_text = ''.join(c for c in sentiment_text if c.isdigit() or c in ['-', '.'])
            
            try:
                sentiment_score = float(sentiment_text)
                # Ensure score is within valid range
                sentiment_score = max(-10, min(10, sentiment_score))
                return sentiment_score
            except ValueError:
                logger.warning(f"Could not parse sentiment score from response: {response.text}")
                if attempt < retry_count - 1:
                    continue
                else:
                    st.warning(f"Could not parse sentiment score for headline: {headline[:50]}...")
                    return None
                
        except Exception as e:
            logger.error(f"Error on attempt {attempt+1}/{retry_count} analyzing sentiment: {str(e)}")
            if attempt < retry_count - 1:
                # Exponential backoff
                wait_time = backoff_factor ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                st.warning(f"Failed to analyze sentiment after {retry_count} attempts. Error: {str(e)}")
                return None

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
