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
from newspaper import Article

# --- Load Environment Variables ---
load_dotenv()

# --- Set up Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Setup Google Gemini API ---
def setup_gemini_api():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")
        return None
    return api_key

# --- Sentiment Analysis ---
def gemini_analyze_sentiment(text: str, api_key: Optional[str]) -> float:
    if not api_key:
        st.error("Gemini API Key not configured.")
        return 0.0
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""Please analyze the sentiment of the following headline and return ONLY integer 
        between -10 (very negative) and 10 (very positive). The number should reflect the sentiment score.
        No explanation, just a number.

        Headline: {text}
        """
        response = model.generate_content(prompt)
        try:
            sentiment_score = float(response.text.strip())
            sentiment_score = max(-10, min(10, sentiment_score))
            return sentiment_score
        except ValueError:
            st.warning(f"Could not convert sentiment response to number: {response.text}")
            return 0.0
    except Exception as e:
        st.error(f"Error with Gemini API: {str(e)}")
        return 0.0

# --- Generate Summary for Full Article ---
def gemini_generate_summary(text: str, api_key: Optional[str]) -> str:
    if not api_key:
        return "API Key not found."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        Summarize the following article content in 3–5 concise bullet points. 
        Avoid speculation, and focus on factual reporting. 

        Article Content:
        {text}
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# --- Calculate Average Sentiment ---
def calculate_average_sentiment(scores):
    valid_scores = [score for score in scores if score is not None]
    if not valid_scores:
        return 0
    return sum(valid_scores) / len(valid_scores)
from datetime import datetime

def fetch_gnews_headlines(ticker, from_date=None, to_date=None):
    api_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        st.error("GNEWS_API_KEY not found in .env")
        return []

    base_url = "https://gnews.io/api/v4/search"
    params = {
        "q": ticker,
        "token": api_key,
        "lang": "en",
        "country": "us",
        "max": 10,  # Fetch the most recent 10
        "sortby": "publishedAt"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])

        # --- Domain filter ---
        allowed_domains = [
            "yahoo.com", "bloomberg.com", "wsj.com", "forbes.com", "fortune.com",
            "axios.com", "ft.com", "cnbc.com", "businessinsider.com",
            "marketwatch.com", "seekingalpha.com"
        ]

        def is_allowed(article):
            return any(domain in article.get("url", "") for domain in allowed_domains)

        # --- Date filter ---
        def is_within_date(article):
            if not (from_date and to_date):
                return True  # If no range set, allow all

            pub_date_str = article.get("publishedAt", "")
            try:
                pub_date = datetime.fromisoformat(pub_date_str.rstrip("Z")).date()
                return from_date <= pub_date.isoformat() <= to_date
            except Exception:
                return False

        # --- Final filtered list ---
        filtered = [a for a in articles if is_allowed(a) and is_within_date(a)]

        if not filtered:
            st.warning(f"No GNews articles for '{ticker}' in the selected date range.")
        return filtered

    except Exception as e:
        st.error(f"GNews error: {str(e)}")
        return []

# --- Fetch News Headlines ---
from datetime import datetime
def fetch_news_headlines(ticker, from_date=None, to_date=None):
    all_articles = []

    # --- NewsAPI ---
    newsapi_key = os.getenv("NEWSAPI_KEY")
    if newsapi_key:
        newsapi_url = (
            f"https://newsapi.org/v2/everything?q={ticker}"
            f"&domains=yahoo.com,bloomberg.com,wsj.com,forbes.com,fortune.com,axios.com,ft.com,cnbc.com,"
            f"businessinsider.com,marketwatch.com,seekingalpha.com"
            f"&apiKey={newsapi_key}&pageSize=10&sortBy=publishedAt&language=en"
        )

        try:
            response = requests.get(newsapi_url)
            response.raise_for_status()
            news_data = response.json()
            articles = news_data.get("articles", [])

            # --- Date filter for NewsAPI articles ---
            def is_within_date(article):
                if not (from_date and to_date):
                    return True
                pub_date_str = article.get("publishedAt", "")
                try:
                    pub_date = datetime.fromisoformat(pub_date_str.rstrip("Z")).date()
                    return from_date <= pub_date.isoformat() <= to_date
                except Exception:
                    return False

            filtered_articles = [a for a in articles if is_within_date(a)]
            all_articles.extend(filtered_articles)

        except Exception as e:
            st.warning(f"NewsAPI error: {str(e)}")
    else:
        st.warning("NEWSAPI_KEY not found. Skipping NewsAPI.")

    # --- GNews ---
    gnews_articles = fetch_gnews_headlines(ticker, from_date, to_date)
    all_articles.extend(gnews_articles)

    # --- Final sort: Most recent first ---
    def get_date(article):
        try:
            return datetime.fromisoformat(article.get("publishedAt", "").rstrip("Z"))
        except Exception:
            return datetime.min

    all_articles.sort(key=get_date, reverse=True)

    if not all_articles:
        st.warning(f"No news found for ticker: {ticker} in the selected date range.")
    return all_articles



# --- Fetch Full Article Text ---
def fetch_full_article_text(url: str) -> Optional[str]:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logger.error(f"Failed to fetch article from {url}: {e}")
        return None

# --- Process Each Article ---
def process_article(row, api_key):
    headline = row['headline']
    url = row['url']
    try:
        article_text = fetch_full_article_text(url)
        if article_text and len(article_text.strip()) > 100:
            summary = gemini_generate_summary(article_text, api_key)
            prompt = f"""
            Analyze the overall sentiment of this news.
            Headline: {headline}
            Summary: {summary}
            Provide a single sentiment score from -10 (very negative) to 10 (very positive). No explanation.
            """
        else:
            raise ValueError("Article too short or empty")
    except Exception as e:
        logger.warning(f"Fallback to headline for sentiment: {url} | Reason: {e}")
        summary = "Summary unavailable."
        prompt = f"""
        Analyze the sentiment of this headline:

        {headline}

        Give a score from -10 (very negative) to 10 (very positive). No explanation.
        """
    sentiment_score = gemini_analyze_sentiment(prompt, api_key)
    return summary, sentiment_score

# --- Streamlit App ---
st.set_page_config(
    page_title="Stock News Sentiment Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("Stock News Sentiment Analysis")
st.markdown("""
This application analyzes sentiment for the latest news headlines related to a stock ticker.
Enter a stock ticker symbol (e.g., AAPL, TSLA, GOOGL) to get started.
""")

ticker_input = st.text_input("Enter Stock Ticker Symbol:", placeholder="e.g., AAPL")
today = datetime.today()
default_start = today - timedelta(days=7)
start_date = st.date_input("Start Date", default_start)
end_date = st.date_input("End Date", today)

if ticker_input:
    ticker = ticker_input.strip().upper()
    with st.spinner(f"Fetching news headlines for {ticker}..."):
        try:
            headlines = fetch_news_headlines(ticker, from_date=start_date.strftime("%Y-%m-%d"), to_date=end_date.strftime("%Y-%m-%d"))
            if not headlines:
                st.error(f"No news headlines found for ticker {ticker}. Please check if the ticker is valid.")
            else:
                df = pd.DataFrame({
                    'headline': [h['title'] for h in headlines],
                    'url': [h['url'] for h in headlines],
                    'source': [h['source']['name'] for h in headlines],
                    'publishedAt': [h['publishedAt'] for h in headlines]
                })
                st.subheader(f"Analyzing sentiment for {ticker} headlines...")
                progress_bar = st.progress(0)
                api_key = setup_gemini_api()
                df['summary'] = None
                df['combined_sentiment'] = None
                for i, row in df.iterrows():
                    with st.spinner(f"Processing {i+1}/{len(df)}"):
                        summary, sentiment = process_article(row, api_key)
                        df.at[i, 'summary'] = summary
                        df.at[i, 'combined_sentiment'] = sentiment
                        progress_bar.progress((i + 1) / len(df))
                avg_sentiment = calculate_average_sentiment(df['combined_sentiment'])
                valid_scores_count = df['combined_sentiment'].notnull().sum()
                st.success(f"Analysis completed! Analyzed {valid_scores_count} out of {len(df)} headlines.")

                # Display gauge + metrics
                st.subheader("Overall Sentiment Analysis")
                col1, col2 = st.columns([1, 3])
                with col1:
                    if avg_sentiment > 0:
                        st.markdown("### 📈 Positive")
                        sentiment_color = "green"
                    elif avg_sentiment < 0:
                        st.markdown("### 📉 Negative")
                        sentiment_color = "red"
                    else:
                        st.markdown("### ⚖️ Neutral")
                        sentiment_color = "gray"
                    st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")
                with col2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=avg_sentiment,
                        title={'text': f"Sentiment Score for {ticker}"},
                        gauge={
                            'axis': {'range': [-10, 10]},
                            'bar': {'color': sentiment_color},
                            'steps': [
                                {'range': [-10, -5], 'color': 'rgba(255, 0, 0, 0.3)'},
                                {'range': [-5, 0], 'color': 'rgba(255, 165, 0, 0.3)'},
                                {'range': [0, 5], 'color': 'rgba(144, 238, 144, 0.3)'},
                                {'range': [5, 10], 'color': 'rgba(0, 128, 0, 0.3)'}
                            ]
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Analyzed Headlines")
                display_df = df.dropna(subset=['combined_sentiment']).copy()
                if display_df.empty:
                    st.warning("No headlines with valid sentiment scores to display.")
                else:
                    display_df['sentiment_category'] = display_df['combined_sentiment'].apply(
                        lambda x: "Positive" if x > 0 else ("Neutral" if x == 0 else "Negative")
                    )
                    display_df['headline_with_link'] = display_df.apply(
                        lambda row: f"<a href='{row['url']}' target='_blank'>{row['headline']}</a>", axis=1
                    )

                    with st.expander("View All Headlines", expanded=True):
                        for i, row in display_df.iterrows():
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(row['headline_with_link'], unsafe_allow_html=True)
                                st.markdown(f"**Summary:** {row['summary']}")
                                st.caption(f"Source: {row['source']} | Published: {row['publishedAt']}")
                            with col2:
                                score = row['combined_sentiment']
                                if score > 3:
                                    st.markdown(f"#### 😀 {score:.1f}")
                                elif score > 0:
                                    st.markdown(f"#### 🙂 {score:.1f}")
                                elif score == 0:
                                    st.markdown(f"#### 😐 {score:.1f}")
                                elif score > -3:
                                    st.markdown(f"#### 🙁 {score:.1f}")
                                else:
                                    st.markdown(f"#### 😞 {score:.1f}")
                            st.divider()

                    # Visualizations
                    st.subheader("Sentiment Visualizations")
                    col1, col2 = st.columns(2)
                    with col1:
                        display_df['headline_short'] = display_df['headline'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
                        bar_fig = px.bar(
                            display_df,
                            x='headline_short',
                            y='combined_sentiment',
                            color='combined_sentiment',
                            color_continuous_scale='RdYlGn',
                            labels={'headline_short': 'Headline', 'combined_sentiment': 'Sentiment Score'},
                            title=f'Sentiment Scores for {ticker} Headlines'
                        )
                        bar_fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(bar_fig, use_container_width=True)
                    with col2:
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
                    hist_fig = px.histogram(
                        display_df,
                        x='combined_sentiment',
                        nbins=20,
                        color_discrete_sequence=['lightblue'],
                        labels={'combined_sentiment': 'Sentiment Score'},
                        title=f'Distribution of Sentiment Scores for {ticker}'
                    )
                    hist_fig.add_vline(x=avg_sentiment, line_dash="dash", line_color="red", annotation_text=f"Avg: {avg_sentiment:.2f}")
                    hist_fig.update_layout(bargap=0.1)
                    st.plotly_chart(hist_fig, use_container_width=True)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)

st.markdown("---")
st.caption("Stock News Sentiment Analysis Application | Data is fetched in real-time")
