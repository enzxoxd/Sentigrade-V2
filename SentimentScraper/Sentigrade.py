from datetime import datetime, timedelta
import os
import requests
from bs4 import BeautifulSoup
import re
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from typing import Optional, Tuple
from newspaper import Article
import plotly.express as px
import logging
import google.generativeai as genai
from langdetect import detect
from concurrent.futures import ThreadPoolExecutor
from dateutil.parser import parse
import validators
import sqlite3
from uuid import uuid4
import numpy as np
import os
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
import logging

# Debug: Check if db_utils.py exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_UTILS_PATH = os.path.join(BASE_DIR, 'db_utils.py')
if not os.path.exists(DB_UTILS_PATH):
    st.error(f"db_utils.py not found at {DB_UTILS_PATH}. Please create it in the project root.")
    st.stop()

try:
    from db_utils import init_db
except ImportError as e:
    st.error(f"Failed to import db_utils from {DB_UTILS_PATH}. Ensure db_utils.py is correctly named and in the project root.")
    raise ImportError(f"Cannot import init_db: {e}")


# --- Load environment variables ---
load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize database ---
try:
    init_db()
except Exception as e:
    st.error(f"Failed to initialize database: {e}")
    st.stop()
# --- Streamlit page config ---
st.set_page_config(page_title="Sentigrade", page_icon="📈", layout="wide")
st.title("📊 Sentigrade")


# --- Helper Functions ---
def parse_relative_time(time_text):
    now = datetime.now()
    try:
        if "ago" in time_text.lower():
            match = re.search(r"(\d+)\s+(\w+)", time_text)
            if match:
                num, unit = int(match.group(1)), match.group(2).lower()
                if "min" in unit:
                    return now - timedelta(minutes=num)
                elif "hour" in unit:
                    return now - timedelta(hours=num)
                elif "day" in unit:
                    return now - timedelta(days=num)
        elif "just now" in time_text.lower():
            return now
        elif re.match(r"\w+ \d{1,2}, \d{4}", time_text):
            return datetime.strptime(time_text, "%B %d, %Y")
    except Exception as e:
        logger.warning(f"Time parse error: {str(e)}")
    return None

@st.cache_data(ttl=3600)
def fetch_yahoo_news(ticker, limit=3):
    url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    articles = []

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        anchors = soup.select("a.subtle-link")
        print(f"[DEBUG] Found {len(anchors)} article blocks")

        for anchor in anchors:
            headline_tag = anchor.find("h3")
            desc_tag = anchor.find("p")
            parent = anchor.parent

            if not headline_tag:
                continue

            # Title, Description, URL
            title = headline_tag.text.strip()
            description = desc_tag.text.strip() if desc_tag else ""
            link = anchor["href"]
            full_url = link if link.startswith("http") else f"https://finance.yahoo.com{link}"

            # Time + Source
            publishing_div = parent.find("div", class_="publishing") if parent else None
            source, time_str = "Yahoo Finance", "unknown"
            published_at = None

            if publishing_div:
                parts = publishing_div.text.strip().split("•")
                if len(parts) == 2:
                    source = parts[0].strip()
                    time_str = parts[1].strip()
                    published_at = parse_relative_time(time_str)

            articles.append({
                "title": title,
                "url": full_url,
                "description": description,
                "publishedAt": published_at.isoformat() if published_at else time_str,
                "source": source,
            })

            if len(articles) >= limit:
                break

        return articles

    except Exception as e:
        print(f"[Scraper Error] {e}")
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
            f"q={ticker}&"
            f"from={from_date}&to={to_date}&"
            f"language=en&"
            f"sortBy=popularity&"
            f"pageSize=1&"
            f"apiKey={api_key}"
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
            logger.warning(f"Error fetching for {from_date}: {e}")

    sorted_articles = sorted(collected_articles, key=lambda x: x['publishedAt'], reverse=True)
    logger.info(f"Fetched {len(sorted_articles)} valid NewsAPI articles for {ticker}")
    return sorted_articles[:limit]

def fetch_full_article_text(url: str) -> Optional[str]:
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
    prompt = f"""
    Summarize this financial news article in 2-3 concise sentences.

    Article:
    {article_text}
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini summarization failed: {e}")
        return "Summary unavailable."

def gemini_analyze_sentiment(headline, summary, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    Analyze the sentiment of this financial news article.

    Headline: {headline}
    Summary: {summary}

    Provide a single sentiment score from -10 (very negative) to 10 (very positive).
    Respond only with the number.
    """
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        match = re.search(r"-?\d+(\.\d+)?", response_text)
        if match:
            return float(match.group())
        else:
            raise ValueError(f"Unexpected response from Gemini: {response_text}")
    except Exception as e:
        logger.error(f"Gemini sentiment analysis failed for headline '{headline}': {e}")
        return 0.0

def calculate_average_sentiment(scores):
    valid_scores = [s for s in scores if isinstance(s, (int, float)) and not np.isnan(s)]
    return sum(valid_scores) / len(valid_scores) if valid_scores else 0

def analyze_headlines(df, api_key):
    """Analyze sentiment for each headline in the DataFrame."""
    if not {'headline', 'url'}.issubset(df.columns):
        logger.error(f"Input DataFrame missing required columns: {df.columns}")
        st.error("Input data missing 'headline' or 'url' columns.")
        return pd.DataFrame({'headline': [], 'url': [], 'summary': [], 'combined_sentiment': []})

    logger.info(f"Input DataFrame shape: {df.shape}, columns: {df.columns}")
    logger.info(f"Input DataFrame head:\n{df.head().to_string()}")
    df = df.copy()
    df['summary'] = None
    df['combined_sentiment'] = 0.0  # Initialize with default value
    processed_rows = 0
    valid_sentiment_scores = 0

    for i, row in df.iterrows():
        headline = row['headline']
        url = row['url']
        if not isinstance(headline, str) or not headline.strip() or not isinstance(url, str) or not url.strip() or not url.startswith('http'):
            logger.warning(f"Skipping row {i}: Invalid headline or URL (headline: {headline}, url: {url})")
            continue

        try:
            article_text = fetch_full_article_text(url)
            if article_text and len(article_text.strip()) > 100:
                summary = gemini_generate_summary(article_text, api_key)
            else:
                raise ValueError("Article too short or empty")
        except Exception as e:
            logger.warning(f"Fallback to headline for sentiment: {url} | Reason: {e}")
            summary = "Summary unavailable."

        sentiment_score = gemini_analyze_sentiment(headline, summary, api_key)
        df.at[i, 'summary'] = summary
        df.at[i, 'combined_sentiment'] = sentiment_score
        processed_rows += 1
        if sentiment_score != 0.0:
            valid_sentiment_scores += 1
            logger.info(f"Row {i}: Headline='{headline}', Sentiment={sentiment_score}")

    logger.info(f"Processed {processed_rows} out of {len(df)} rows for sentiment analysis")
    logger.info(f"Valid sentiment scores (non-zero): {valid_sentiment_scores}")
    logger.info(f"Output DataFrame columns: {df.columns}")
    logger.info(f"Output DataFrame head:\n{df.head().to_string()}")
    logger.info(f"Combined sentiment values: {df['combined_sentiment'].tolist()}")

    if processed_rows == 0:
        logger.error("No rows processed. Check input data (headlines/URLs).")
        st.error("No valid articles processed. Check input data or API connectivity.")
        return pd.DataFrame({'headline': [], 'url': [], 'summary': [], 'combined_sentiment': []})

    if valid_sentiment_scores == 0:
        logger.error("No valid sentiment scores generated. Check Gemini API or article content.")
        st.error("Sentiment analysis failed: No valid sentiment scores generated. Check API key, network, or article content.")
        return pd.DataFrame({'headline': [], 'url': [], 'summary': [], 'combined_sentiment': []})

    return df


    if isinstance(price_data.columns, pd.MultiIndex):
        price_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in price_data.columns]

    close_col = next((col for col in price_data.columns if 'Close' in col), None)
    if not close_col:
        raise KeyError("No 'Close' column found in price data.")

    price_data = price_data.copy()
    price_data.index = pd.to_datetime(price_data.index).normalize()

    sentiment_data = sentiment_data.copy()
    sentiment_data.index = pd.to_datetime(sentiment_data.index).normalize()

    daily_sentiment = sentiment_data.groupby(sentiment_data.index).agg({
        'avg_sentiment': 'mean',
        'article_count': 'sum'
    })

    merged_df = price_data[[close_col]].join(daily_sentiment, how='left')
    merged_df.rename(columns={close_col: 'Close'}, inplace=True)
    merged_df.sort_index(inplace=True)

    merged_df[['avg_sentiment', 'article_count']] = merged_df[['avg_sentiment', 'article_count']].fillna(0)

    return merged_df

def save_to_database(ticker, data_df):
    """Save sentiment and price data to SQLite database"""
    db_path = "stock_sentiment.db"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create table with all required columns, including signal
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_history (
                date TEXT,
                ticker TEXT,
                Close REAL,
                avg_sentiment REAL,
                article_count INTEGER,
                record_date TEXT,
                signal INTEGER
            )
        """)
        conn.commit()

        # Verify table schema
        cursor.execute("PRAGMA table_info(sentiment_history)")
        columns = [info[1] for info in cursor.fetchall()]
        logger.info(f"Table schema: {columns}")

        # If 'signal' column is missing, add it
        if 'signal' not in columns:
            logger.info("Adding 'signal' column to sentiment_history table")
            cursor.execute("ALTER TABLE sentiment_history ADD COLUMN signal INTEGER")
            conn.commit()
            logger.info("Successfully added 'signal' column")

        # Prepare DataFrame for saving
        df_to_save = data_df.copy()
        if df_to_save.index.name is None:
            df_to_save.index.name = 'date'
        df_to_save = df_to_save.reset_index()

        df_to_save['record_date'] = datetime.now().strftime("%Y-%m-%d")
        df_to_save['ticker'] = ticker

        # Ensure only expected columns are saved
        expected_columns = ['date', 'ticker', 'Close', 'avg_sentiment', 'article_count', 'record_date', 'signal']
        df_to_save = df_to_save[[col for col in expected_columns if col in df_to_save.columns]]

        df_to_save.to_sql('sentiment_history', conn, if_exists='append', index=False)
        logger.info(f"Saved {len(df_to_save)} records to database for {ticker}")
    except Exception as e:
        logger.error(f"Database save error: {str(e)}")
        st.error(f"Failed to save to database: {str(e)}")
    finally:
        conn.close()

def load_ticker_history(ticker):
    """Load historical data for a ticker from the database"""
    db_path = "stock_sentiment.db"

    try:
        conn = sqlite3.connect(db_path)
        query = f"""
        SELECT date, Close, avg_sentiment, article_count, signal
        FROM sentiment_history
        WHERE ticker = '{ticker}'
        ORDER BY date
        """

        df = pd.read_sql_query(query, conn, parse_dates=['date'], index_col='date')
        logger.info(f"Loaded {len(df)} records from database for {ticker}")
        return df
    except Exception as e:
        logger.error(f"Database load error: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()


    fast_ma = vbt.MA.run(df['Close'], window=fast_window)
    slow_ma = vbt.MA.run(df['Close'], window=slow_window)
    entries = fast_ma.ma_crossed_above(slow_ma)

    long_entries = entries & (df['avg_sentiment'] > sentiment_threshold)
    exits = fast_ma.ma_crossed_below(slow_ma)

    pf = vbt.Portfolio.from_signals(df['Close'], entries=long_entries, exits=exits, init_cash=10000)

    return pf.stats(), pf.plot()

# --- Streamlit UI ---

if 'ticker' not in st.session_state:
    st.session_state['ticker'] = ''  # Default value

ticker_input = st.text_input("Enter ticker symbol:", value=st.session_state['ticker'], key="main_ticker_input")
st.session_state['ticker'] = ticker_input  # Update the session state with any new value

def run_sentigrade_for_ticker(ticker):
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Gemini API key not found in .env. Please set GEMINI_API_KEY.")
        st.stop()

    with st.spinner(f"Fetching news for {ticker}..."):
        yahoo_articles_all = fetch_yahoo_news(ticker, limit=100)
        yahoo_count = len(yahoo_articles_all)

        if yahoo_count == 0:
            st.warning(f"No Yahoo Finance articles found for {ticker}. Trying alternative sources...")
            newsapi_articles = fetch_newsapi_headlines(ticker, limit=3)
            articles_to_process = newsapi_articles
        else:
            valid_articles = [
                article for article in yahoo_articles_all
                if article.get('title') and article.get('description') and article.get('url')
                and len(article['description']) > 50
                and article['url'].startswith("https://")
            ]
            if len(valid_articles) < 9:
                articles_to_process = valid_articles
            else:
                first_indices = [0, 1, 2]
                mid_point = len(valid_articles) // 2
                middle_indices = [mid_point - 1, mid_point, mid_point + 1]
                last_indices = [len(valid_articles) - 3, len(valid_articles) - 2, len(valid_articles) - 1]
                selected_indices = [i for i in first_indices + middle_indices + last_indices if i < len(valid_articles) and valid_articles[i]['url'].startswith('https://')]
                articles_to_process = [valid_articles[i] for i in selected_indices]

        if not articles_to_process:
            st.error("No valid articles found to process. Please try a different ticker symbol or check API connectivity.")
            st.stop()

        df = pd.DataFrame({
            'headline': [a['title'] for a in articles_to_process],
            'url': [a['url'] for a in articles_to_process],
            'source': [a['source'] for a in articles_to_process],
            'origin': [a.get('origin', '') for a in articles_to_process],
            'publishedAt': [a['publishedAt'] for a in articles_to_process],
            'description': [a['description'] for a in articles_to_process]
        })

        df = df.dropna(subset=['headline', 'url'])
        df = df[df['headline'].str.strip() != '']
        df = df[df['url'].str.startswith('http')]
        if df.empty:
            st.error("No valid articles after validation. Please try a different ticker or check data sources.")
            st.stop()

        df['publishedAt_dt'] = pd.to_datetime(df['publishedAt'], errors='coerce', utc=True)
        df = df.sort_values(by='publishedAt_dt', ascending=False).reset_index(drop=True)

    with st.spinner("Analyzing sentiment. Please do not change tabs..."):
        analyzed_df = analyze_headlines(df.copy(), api_key)
        if 'combined_sentiment' not in analyzed_df.columns:
            st.error("Sentiment analysis failed: 'combined_sentiment' column missing. Check API key or input data.")
            st.stop()
        if analyzed_df.empty or analyzed_df['combined_sentiment'].isna().all():
            st.error("Sentiment analysis failed: No valid sentiment scores generated. Check API key, network, or input data.")
            st.stop()

    st.header(f"Sentiment Results for {ticker}")
    st.dataframe(analyzed_df)
    avg_sentiment = calculate_average_sentiment(analyzed_df['combined_sentiment'].tolist())
    st.write(f"**Average Sentiment:** {avg_sentiment:.2f}")

# --- Core Analysis Function ---
def analyze_ticker(ticker: str):
    ticker = ticker.strip().upper()

    if "previous_ticker" not in st.session_state:
        st.session_state.previous_ticker = ""

    if ticker != st.session_state.previous_ticker:
        st.session_state.analyzed_df = None
        st.session_state.previous_ticker = ticker

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Gemini API key not found in .env. Please set GEMINI_API_KEY.")
        st.stop()

    with st.spinner(f"Fetching news for {ticker}..."):
        yahoo_articles_all = fetch_yahoo_news(ticker, limit=100)
        yahoo_count = len(yahoo_articles_all)

        if yahoo_count == 0:
            st.warning(f"No Yahoo Finance articles found for {ticker}. Trying alternative sources...")
            newsapi_articles = fetch_newsapi_headlines(ticker, limit=3)
            articles_to_process = newsapi_articles
        else:
            valid_articles = [
                article for article in yahoo_articles_all
                if article.get('title') and article.get('description') and article.get('url')
                and len(article['description']) > 50
                and article['url'].startswith("https://")
            ]

            if len(valid_articles) < 9:
                articles_to_process = valid_articles
            else:
                first_indices = [0, 1, 2]
                mid_point = len(valid_articles) // 2
                middle_indices = [mid_point - 1, mid_point, mid_point + 1]
                last_indices = [len(valid_articles) - 3, len(valid_articles) - 2, len(valid_articles) - 1]
                selected_indices = [i for i in first_indices + middle_indices + last_indices if i < len(valid_articles)]
                articles_to_process = [valid_articles[i] for i in selected_indices]

        if not articles_to_process:
            st.error("No valid articles found to process.")
            st.stop()

        df = pd.DataFrame({
            'headline': [a['title'] for a in articles_to_process],
            'url': [a['url'] for a in articles_to_process],
            'source': [a['source'] for a in articles_to_process],
            'origin': [a.get('origin', '') for a in articles_to_process],
            'publishedAt': [a['publishedAt'] for a in articles_to_process],
            'description': [a['description'] for a in articles_to_process]
        })

        df = df.dropna(subset=['headline', 'url'])
        df = df[df['headline'].str.strip() != '']
        df = df[df['url'].str.startswith('http')]
        df['publishedAt_dt'] = pd.to_datetime(df['publishedAt'], errors='coerce', utc=True)
        df = df.sort_values(by='publishedAt_dt', ascending=False).reset_index(drop=True)

    if st.session_state.analyzed_df is None:
        with st.spinner("Analyzing sentiment. Please wait..."):
            analyzed_df = analyze_headlines(df.copy(), api_key)

            if 'combined_sentiment' not in analyzed_df.columns or analyzed_df['combined_sentiment'].isna().all():
                st.error("Sentiment analysis failed. Please check API key or network.")
                st.stop()

            st.session_state.analyzed_df = analyzed_df
    else:
        analyzed_df = st.session_state.analyzed_df

    avg_sentiment = calculate_average_sentiment(analyzed_df['combined_sentiment'])
    st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")

    analyzed_df['sentiment_category'] = analyzed_df['combined_sentiment'].apply(
        lambda x: "Positive" if x > 0 else "Neutral" if x == 0 else "Negative"
    )

    sentiment_by_date = analyzed_df.copy()
    sentiment_by_date['date'] = pd.to_datetime(sentiment_by_date['publishedAt'], errors='coerce').dt.normalize()
    daily_sentiment = sentiment_by_date.groupby('date').agg(
        avg_sentiment=('combined_sentiment', 'mean'),
        article_count=('headline', 'count')
    ).reset_index()
    daily_sentiment.set_index('date', inplace=True)

    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        bar_fig = px.bar(
            analyzed_df.assign(short_headline=analyzed_df['headline'].apply(lambda x: ' '.join(x.split()[:4]) + "...")),
            x='short_headline',
            y='combined_sentiment',
            color='combined_sentiment',
            color_continuous_scale='RdYlGn',
            title='Headline Sentiment Scores'
        )
        bar_fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(bar_fig, use_container_width=True)

    with col2:
        pie_fig = px.pie(
            analyzed_df,
            names='sentiment_category',
            title="Sentiment Distribution",
            color='sentiment_category',
            color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"}
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    # Summaries
    st.subheader(f"📰 Headlines for {ticker}")
    for _, row in analyzed_df.iterrows():
        st.markdown(f"**[{row['headline']}]({row['url']})**")
        st.markdown(f"*{row.get('summary', row.get('description', ''))}*")
        st.caption(f"Source: {row['source']} ({row['origin']}) | Published: {row['publishedAt']} | Sentiment Score: {row['combined_sentiment']}")
        st.divider()

    # Download
    st.download_button(
        label="⬇️ Download CSV",
        data=analyzed_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_sentiment.csv",
        mime="text/csv"
    )

    # Re-analyze
    if st.button(f"🔁 Re-analyze {ticker}"):
        with st.spinner("Re-analyzing..."):
            st.session_state.analyzed_df = None
            st.experimental_rerun()


# --- Run Single or Batch ---
if ticker_input:
    st.session_state['ticker'] = ticker_input.strip().upper()
    analyze_ticker(st.session_state['ticker'])
elif 'ticker' in st.session_state:
    analyze_ticker(st.session_state['ticker'])


st.markdown("### 🚀 Or analyze a batch of popular tickers:")
batch_tickers = ['SPY', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META']
if st.button("Analyze Top 6 Tickers"):
    for ticker in batch_tickers:
        st.markdown(f"## {ticker}")
        analyze_ticker(ticker)
        st.markdown("---")

st.markdown("---")
st.caption("2025 Stock Sentiment & Trading AI | Powered by Yahoo Finance + NewsAPI + Gemini")
