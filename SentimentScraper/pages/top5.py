from datetime import datetime, timedelta
import os
import requests
from bs4 import BeautifulSoup
import re
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from newspaper import Article
import logging
import google.generativeai as genai
import sqlite3
import altair as alt

# --- Load environment variables ---
load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit Page Config ---
st.set_page_config(page_title="Sentigrade", page_icon="📈", layout="wide")
st.title("📊 Sentigrade: Top 5 SPY Ticker Sentiment Tracker")

# --- Top Tickers ---
TOP_5_SPY_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]

# --- Helper Functions ---

@st.cache_data(ttl=3600)
def fetch_yahoo_news(ticker, limit=10):
    url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    articles = []
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        elements = soup.select("li.js-stream-content a")
        for element in elements:
            headline_tag = element.find("h3")
            if not headline_tag or not headline_tag.text.strip():
                continue
            title = headline_tag.text.strip()
            link = element["href"]
            url_full = link if link.startswith("http") else f"https://finance.yahoo.com{link}"
            articles.append({
                "title": title,
                "url": url_full,
                "publishedAt": None,
                "source": {"name": "Yahoo Finance"},
                "description": "",
                "origin": "Yahoo"
            })
            if len(articles) >= limit:
                break
        return articles
    except Exception as e:
        logger.error(f"Yahoo scraping error: {str(e)}")
        return []

def fetch_newsapi_headlines(ticker, limit=3):
    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        st.error("Missing NEWSAPI_KEY in .env or environment")
        return []
    collected_articles = []
    headers = {"Authorization": api_key}
    for day_offset in range(3):
        date = datetime.utcnow() - timedelta(day_offset)
        from_date = date.strftime("%Y-%m-%d")
        to_date = from_date
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={ticker}&from={from_date}&to={to_date}&language=en&sortBy=popularity&pageSize=1&apiKey={api_key}"
        )
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            article_list = data.get("articles", [])
            if article_list:
                article = article_list[0]
                published_date = article.get("publishedAt", "").split("T")[0]
                title = article.get("title", "")
                url = article.get("url", "")
                description = article.get("description", "")
                if title and url and description and url.startswith("http"):
                    collected_articles.append({
                        "title": title,
                        "description": description,
                        "url": url,
                        "publishedAt": published_date,
                        "source": {"name": article.get("source", {}).get("name", "NewsAPI")},
                        "origin": "NewsAPI"
                    })
        except Exception as e:
            logger.warning(f"Error fetching NewsAPI for {from_date}: {e}")
    sorted_articles = sorted(collected_articles, key=lambda x: x['publishedAt'], reverse=True)
    return sorted_articles[:limit]

def fetch_full_article_text(url: str):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logger.error(f"Failed to fetch article from {url}: {e}")
        return None

def gemini_generate_summary(article_text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""Summarize this financial news article in 2-3 concise sentences.\n\nArticle:\n{article_text}"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini summarization failed: {e}")
        return "Summary unavailable."

def gemini_analyze_sentiment(headline, summary, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""Analyze the sentiment of this financial news article.\n\nHeadline: {headline}\nSummary: {summary}\n\nProvide a single sentiment score from -10 (very negative) to 10 (very positive). Respond only with the number."""
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        match = re.search(r"-?\d+(\.\d+)?", response_text)
        if match:
            return float(match.group())
        else:
            raise ValueError(f"Unexpected response: {response_text}")
    except Exception as e:
        logger.error(f"Gemini sentiment analysis failed for '{headline}': {e}")
        return 0.0

def analyze_headlines(df, api_key):
    df = df.copy()
    df['summary'] = None
    df['combined_sentiment'] = 0.0
    for i, row in df.iterrows():
        headline = row['headline']
        url = row['url']
        try:
            article_text = fetch_full_article_text(url)
            summary = gemini_generate_summary(article_text, api_key) if article_text and len(article_text) > 100 else "Summary unavailable."
        except:
            summary = "Summary unavailable."
        sentiment_score = gemini_analyze_sentiment(headline, summary, api_key)
        df.at[i, 'summary'] = summary
        df.at[i, 'combined_sentiment'] = sentiment_score
    return df

@st.cache_data(ttl=3600)
def fetch_and_analyze_for_ticker(ticker, api_key):
    yahoo_news = fetch_yahoo_news(ticker)
    newsapi_news = fetch_newsapi_headlines(ticker)
    combined_news = yahoo_news + newsapi_news
    if not combined_news:
        return pd.DataFrame()
    news_df = pd.DataFrame(combined_news)
    news_df.rename(columns={'title': 'headline'}, inplace=True)
    news_df['ticker'] = ticker
    analyzed_df = analyze_headlines(news_df, api_key)
    analyzed_df['ticker'] = ticker
    return analyzed_df

def save_to_database(ticker, data_df):
    db_path = "stock_sentiment.db"
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_history (
                date TEXT,
                ticker TEXT,
                avg_sentiment REAL,
                article_count INTEGER,
                record_date TEXT,
                PRIMARY KEY (date, ticker)
            )
        """)
        conn.commit()
        for _, row in data_df.iterrows():
            date_str = row['date'].strftime("%Y-%m-%d")
            cursor.execute("SELECT COUNT(*) FROM sentiment_history WHERE ticker = ? AND date = ?", (ticker, date_str))
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    INSERT INTO sentiment_history (date, ticker, avg_sentiment, article_count, record_date)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    date_str, ticker,
                    row['avg_sentiment'],
                    row['article_count'],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
        conn.commit()
        conn.close()
        logger.info(f"Saved sentiment data for {ticker} to database.")
    except Exception as e:
        logger.error(f"Failed to save to database: {e}")

def load_sentiment_history():
    try:
        conn = sqlite3.connect("stock_sentiment.db")
        df = pd.read_sql("SELECT * FROM sentiment_history", conn)
        conn.close()
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(by=['ticker', 'date'], inplace=True)
        df['rolling_avg_sentiment'] = df.groupby('ticker')['avg_sentiment'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        return df
    except Exception as e:
        logger.warning(f"Could not load historical data: {e}")
        return pd.DataFrame()

# --- Main App ---
def main():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Missing GEMINI_API_KEY in environment variables.")
        return

    dfs = []
    with st.spinner("Fetching and analyzing news for top 5 SPY tickers..."):
        for ticker in TOP_5_SPY_TICKERS:
            st.info(f"Processing {ticker}...")
            df = fetch_and_analyze_for_ticker(ticker, api_key)
            if not df.empty:
                dfs.append(df)

    if not dfs:
        st.warning("No news articles or sentiment data found for the selected tickers.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['publishedAt'] = pd.to_datetime(combined_df['publishedAt'], errors='coerce')

    # Combine and display sentiment data
    st.subheader("📰 Sentiment Data Table")
    st.dataframe(
        combined_df[['publishedAt', 'ticker', 'headline', 'combined_sentiment', 'summary', 'url']]
        .sort_values(by=['ticker', 'publishedAt'], ascending=[True, False]),
        use_container_width=True
    )

    st.subheader("📈 Sentiment Over Time (Top 5 SPY Tickers)")

    combined_df['date'] = combined_df['publishedAt'].dt.date
    daily_sentiment = combined_df.groupby(['date', 'ticker']).agg(
        avg_sentiment=('combined_sentiment', 'mean'),
        article_count=('headline', 'count')
    ).reset_index()

    # Compute aggregate average sentiment per date
    aggregate_sentiment = daily_sentiment.groupby('date').agg(
        avg_sentiment=('avg_sentiment', 'mean'),
        article_count=('article_count', 'sum')
    ).reset_index()

    # Altair chart: Sentiment over time
    base_chart = alt.Chart(daily_sentiment).mark_line().encode(
        x='date:T',
        y='avg_sentiment:Q',
        color='ticker:N',
        tooltip=['date:T', 'avg_sentiment:Q', 'ticker:N']
    ).properties(
        title="Sentiment Over Time for Top 5 SPY Tickers"
    )

    # Aggregated black dashed line for overall sentiment
    aggregated_line = alt.Chart(aggregate_sentiment).mark_line(strokeDash=[5, 5], color='black').encode(
        x='date:T',
        y='avg_sentiment:Q',
        tooltip=['date:T', 'avg_sentiment:Q']
    )

    # Combine the chart and the black dashed line
    final_chart = base_chart + aggregated_line

    st.altair_chart(final_chart, use_container_width=True)

    # Save data to SQLite for future use
    for ticker in TOP_5_SPY_TICKERS:
        ticker_df = daily_sentiment[daily_sentiment['ticker'] == ticker]
        save_to_database(ticker, ticker_df)

if __name__ == "__main__":
    main()
