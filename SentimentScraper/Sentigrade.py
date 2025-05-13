from datetime import datetime, timedelta
import os
import requests
from bs4 import BeautifulSoup
import re
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from typing import Optional
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
def fetch_yahoo_news(ticker, limit=1):
    url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    articles = []

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        anchors = soup.select("a.subtle-link")
        logger.debug(f"Found {len(anchors)} article blocks for {ticker}")

        for anchor in anchors:
            headline_tag = anchor.find("h3")
            desc_tag = anchor.find("p")
            parent = anchor.parent

            if not headline_tag:
                continue

            title = headline_tag.text.strip()
            description = desc_tag.text.strip() if desc_tag else ""
            link = anchor["href"]
            full_url = link if link.startswith("http") else f"https://finance.yahoo.com{link}"

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
        logger.error(f"Scraper error for {ticker}: {e}")
        return []

def fetch_newsapi_headlines(ticker, limit=1):
    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        st.error("Missing NEWSAPI_KEY in .env or environment")
        return []

    collected_articles = []
    headers = {"Authorization": api_key}

    for day_offset in range(3):
        date = datetime.utcnow() - timedelta(days=day_offset)
        from_date = date.strftime("%Y-%m;%d")
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
            logger.warning(f"Error fetching NewsAPI for {ticker} on {from_date}: {e}")

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
    if not {'headline', 'url'}.issubset(df.columns):
        logger.error(f"Input DataFrame missing required columns: {df.columns}")
        st.error("Input data missing 'headline' or 'url' columns.")
        return pd.DataFrame({'headline': [], 'url': [], 'summary': [], 'combined_sentiment': []})

    logger.info(f"Input DataFrame shape: {df.shape}, columns: {df.columns}")
    df = df.copy()
    df['summary'] = None
    df['combined_sentiment'] = 0.0
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

    if processed_rows == 0:
        logger.error("No rows processed. Check input data (headlines/URLs).")
        st.error("No valid articles processed. Check input data or API connectivity.")
        return pd.DataFrame({'headline': [], 'url': [], 'summary': [], 'combined_sentiment': []})

    if valid_sentiment_scores == 0:
        logger.error("No valid sentiment scores generated. Check Gemini API or article content.")
        st.error("Sentiment analysis failed: No valid sentiment scores generated. Check API key, network, or article content.")
        return pd.DataFrame({'headline': [], 'url': [], 'summary': [], 'combined_sentiment': []})

    return df

def save_to_database(ticker, data_df, session_id=None):
    if not ticker or not isinstance(ticker, str) or not ticker.strip():
        logger.error(f"No valid ticker symbol provided for saving to database, session_id={session_id}, session_state.ticker={st.session_state.get('ticker', 'None')}")
        st.error("Cannot save to database: No valid ticker symbol provided")
        return False

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
                signal INTEGER,
                session_id TEXT
            )
        """)
        cursor.execute("PRAGMA table_info(sentiment_history)")
        columns = [info[1] for info in cursor.fetchall()]
        logger.info(f"Table schema for {ticker}: {columns}")
        if 'session_id' not in columns:
            logger.info("Adding 'session_id' column to sentiment_history table")
            cursor.execute("ALTER TABLE sentiment_history ADD COLUMN session_id TEXT")
            conn.commit()
        df_to_save = data_df.copy()
        if df_to_save.index.name is None:
            df_to_save.index.name = 'date'
        df_to_save = df_to_save.reset_index()
        df_to_save['record_date'] = datetime.now().strftime("%Y-%m-%d")
        df_to_save['ticker'] = ticker
        df_to_save['session_id'] = session_id
        expected_columns = ['date', 'ticker', 'avg_sentiment', 'article_count', 'record_date', 'signal', 'session_id']
        df_to_save = df_to_save[[col for col in expected_columns if col in df_to_save.columns]]
        logger.info(f"Saving {len(df_to_save)} records for {ticker}, session_id={session_id}, columns={df_to_save.columns.tolist()}")
        df_to_save.to_sql('sentiment_history', conn, if_exists='append', index=False)
        conn.commit()
        # Verify save
        cursor.execute("SELECT COUNT(*) FROM sentiment_history WHERE ticker = ? AND session_id = ?", (ticker, session_id))
        saved_count = cursor.fetchone()[0]
        logger.info(f"Verified: {saved_count} records saved for {ticker} with session_id {session_id}")
        return saved_count > 0
    except Exception as e:
        logger.error(f"Database save error for {ticker}: {str(e)}")
        st.error(f"Failed to save to database for {ticker}: {str(e)}")
        return False
    finally:
        conn.close()

def save_batch_to_database(tickers, data_dfs, session_id):
    db_path = "stock_sentiment.db"
    saved_tickers = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_session (
                session_id TEXT PRIMARY KEY,
                created_at TEXT,
                ticker_count INTEGER
            )
        """)
        cursor.execute("PRAGMA table_info(batch_session)")
        columns = [info[1] for info in cursor.fetchall()]
        logger.info(f"Batch session table schema: {columns}")

        # Save batch session metadata
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO batch_session (session_id, created_at, ticker_count) VALUES (?, ?, ?)",
            (session_id, created_at, len(tickers))
        )
        conn.commit()
        logger.info(f"Saved batch session metadata: session_id={session_id}, ticker_count={len(tickers)}")

        # Save sentiment data for each ticker
        for i, (ticker, data_df) in enumerate(zip(tickers, data_dfs)):
            logger.info(f"Processing ticker {ticker} ({i+1}/{len(tickers)}), data_df shape={data_df.shape}, rows={len(data_df)}")
            if not data_df.empty:
                df_to_save = data_df.copy()
                if df_to_save.index.name is None:
                    df_to_save.index.name = 'date'
                df_to_save = df_to_save.reset_index()
                df_to_save['record_date'] = datetime.now().strftime("%Y-%m-%d")
                df_to_save['ticker'] = ticker
                df_to_save['session_id'] = session_id
                expected_columns = ['date', 'ticker', 'avg_sentiment', 'article_count', 'record_date', 'signal', 'session_id']
                df_to_save = df_to_save[[col for col in expected_columns if col in df_to_save.columns]]
                logger.info(f"Saving {len(df_to_save)} records for {ticker}, session_id={session_id}, columns={df_to_save.columns.tolist()}")
                df_to_save.to_sql('sentiment_history', conn, if_exists='append', index=False)
                conn.commit()
                # Verify save
                cursor.execute("SELECT COUNT(*) FROM sentiment_history WHERE ticker = ? AND session_id = ?", (ticker, session_id))
                saved_count = cursor.fetchone()[0]
                if saved_count > 0:
                    saved_tickers.append(ticker)
                    logger.info(f"Verified: {saved_count} records saved for {ticker} with session_id {session_id}")
                else:
                    logger.warning(f"No records saved for {ticker} in batch session {session_id}")
            else:
                logger.warning(f"No sentiment data for {ticker} in batch session {session_id}")
                st.warning(f"No sentiment data available for {ticker}, nothing saved to database")

        # Verify all saves
        cursor.execute("SELECT ticker, COUNT(*) FROM sentiment_history WHERE session_id = ? GROUP BY ticker", (session_id,))
        saved_records = cursor.fetchall()
        logger.info(f"Batch save verification for session {session_id}: {saved_records}")
        st.info(f"Batch save verification: {saved_records}")
        return saved_tickers
    except Exception as e:
        logger.error(f"Batch database save error for session {session_id}: {str(e)}")
        st.error(f"Failed to save batch session {session_id}: {str(e)}")
        return []
    finally:
        conn.close()

def load_batch_session(session_id):
    db_path = "stock_sentiment.db"
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM batch_session WHERE session_id = ?", (session_id,))
        session_data = cursor.fetchone()
        if not session_data:
            logger.error(f"No batch session found for session_id {session_id}")
            return None, pd.DataFrame()

        query = """
        SELECT date, ticker, avg_sentiment, article_count, record_date, signal, session_id
        FROM sentiment_history
        WHERE session_id = ?
        ORDER BY ticker, date
        """
        df = pd.read_sql_query(query, conn, params=(session_id,), parse_dates=['date'], index_col='date')
        logger.info(f"Loaded {len(df)} records for batch session {session_id}, tickers: {df['ticker'].unique().tolist()}")
        return session_data, df
    except Exception as e:
        logger.error(f"Error loading batch session {session_id}: {str(e)}")
        return None, pd.DataFrame()
    finally:
        conn.close()

def load_ticker_history(ticker):
    db_path = "stock_sentiment.db"
    try:
        conn = sqlite3.connect(db_path)
        query = f"""
        SELECT date, avg_sentiment, article_count, signal, session_id
        FROM sentiment_history
        WHERE ticker = '{ticker}'
        ORDER BY date
        """
        df = pd.read_sql_query(query, conn, parse_dates=['date'], index_col='date')
        logger.info(f"Loaded {len(df)} records from database for {ticker}")
        return df
    except Exception as e:
        logger.error(f"Database load error for {ticker}: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def analyze_ticker(ticker: str, is_batch=False):
    ticker = ticker.strip().upper()
    if not ticker:
        st.error("No valid ticker symbol provided for analysis")
        return None

    if is_batch:
        if 'batch_tickers' not in st.session_state:
            st.session_state['batch_tickers'] = []
        if ticker not in st.session_state['batch_tickers']:
            st.session_state['batch_tickers'].append(ticker)
        logger.info(f"Batch ticker added: {ticker}, batch_tickers={st.session_state['batch_tickers']}")

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
        yahoo_articles_all = fetch_yahoo_news(ticker, limit=1)
        yahoo_count = len(yahoo_articles_all)

        if yahoo_count == 0:
            st.warning(f"No Yahoo Finance articles found for {ticker}. Trying alternative sources...")
            newsapi_articles = fetch_newsapi_headlines(ticker, limit=1)
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
            st.error(f"No valid articles found to process for {ticker}.")
            return None

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
        with st.spinner(f"Analyzing sentiment for {ticker}..."):
            analyzed_df = analyze_headlines(df.copy(), api_key)

            if 'combined_sentiment' not in analyzed_df.columns or analyzed_df['combined_sentiment'].isna().all():
                st.error(f"Sentiment analysis failed for {ticker}. Please check API key or network.")
                return None

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
    logger.info(f"Generated daily_sentiment for {ticker}: {len(daily_sentiment)} records")

    if not is_batch:
        try:
            if not daily_sentiment.empty:
                save_to_database(ticker, daily_sentiment)
            else:
                logger.warning(f"No sentiment data to save for {ticker}")
                st.warning(f"No sentiment data available for {ticker}, nothing saved to database")
        except Exception as e:
            logger.error(f"Failed to save sentiment data for {ticker}: {str(e)}")
            st.error(f"Failed to save data for {ticker}: {str(e)}")

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
        st.plotly_chart(bar_fig, use_container_width=True, key=f"bar_chart_{ticker}")

    with col2:
        pie_fig = px.pie(
            analyzed_df,
            names='sentiment_category',
            title="Sentiment Distribution",
            color='sentiment_category',
            color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"}
        )
        st.plotly_chart(pie_fig, use_container_width=True, key=f"pie_chart_{ticker}")

    st.subheader(f"📰 Headlines for {ticker}")
    for _, row in analyzed_df.iterrows():
        st.markdown(f"**[{row['headline']}]({row['url']})**")
        st.markdown(f"*{row.get('summary', row.get('description', ''))}*")
        st.caption(f"Source: {row['source']} ({row['origin']}) | Published: {row['publishedAt']} | Sentiment Score: {row['combined_sentiment']}")
        st.divider()

    st.download_button(
        label="⬇️ Download CSV",
        data=analyzed_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_sentiment.csv",
        mime="text/csv"
    )

    if st.button(f"🔁 Re-analyze {ticker}", key=f"reanalyze_{ticker}"):
        with st.spinner(f"Re-analyzing {ticker}..."):
            st.session_state.analyzed_df = None
            st.experimental_rerun()

    return daily_sentiment

# --- Streamlit UI ---
if 'ticker' not in st.session_state:
    st.session_state['ticker'] = ''
if 'batch_tickers' not in st.session_state:
    st.session_state['batch_tickers'] = []
if 'batch_session_id' not in st.session_state:
    st.session_state['batch_session_id'] = None

popular_tickers = ['SPY', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META']

ticker_input = st.text_input("Enter ticker symbol:", value=st.session_state['ticker'], key="main_ticker_input")
analyze = st.button("Analyze", key="analyze_button")

st.markdown("**Or pick a popular ticker:**")
cols = st.columns(len(popular_tickers))
popular_ticker_clicked = None
for i, ticker in enumerate(popular_tickers):
    if cols[i].button(ticker, key=f"popular_ticker_{ticker}"):
        popular_ticker_clicked = ticker

selected_ticker = None
if analyze and ticker_input:
    selected_ticker = ticker_input.strip().upper()
    st.session_state['ticker'] = selected_ticker
    st.session_state['batch_session_id'] = None
elif popular_ticker_clicked:
    selected_ticker = popular_ticker_clicked
    st.session_state['ticker'] = selected_ticker
    st.session_state['batch_session_id'] = None

if selected_ticker:
    st.success(f"Analyzing data for: {selected_ticker}")
    analyze_ticker(selected_ticker)
else:
    st.info("Please enter a ticker symbol and click Analyze, or select a popular ticker.")

st.markdown("### 🚀 Or analyze a batch of popular tickers:")
if st.button("Analyze Top 6 Tickers", key="batch_analyze"):
    st.session_state['batch_tickers'] = []
    st.session_state['batch_session_id'] = str(uuid4())
    session_id = st.session_state['batch_session_id']
    batch_data = []
    logger.info(f"Starting batch analysis with session_id {session_id}, tickers={popular_tickers}, session_state.ticker={st.session_state.get('ticker', 'None')}")
    for ticker in popular_tickers:
        st.markdown(f"## {ticker}")
        daily_sentiment = analyze_ticker(ticker, is_batch=True)
        if daily_sentiment is not None and not daily_sentiment.empty:
            batch_data.append(daily_sentiment)
            logger.info(f"Collected data for {ticker}: {len(daily_sentiment)} records")
        else:
            logger.warning(f"No valid data collected for {ticker}")
            batch_data.append(pd.DataFrame())
        st.markdown("---")

    logger.info(f"Batch data collected: {len(batch_data)} DataFrames for tickers={popular_tickers}")
    for i, (ticker, df) in enumerate(zip(popular_tickers, batch_data)):
        logger.info(f"Ticker {ticker}: {'empty' if df.empty else f'{len(df)} records'}")

    if any(not df.empty for df in batch_data):
        saved_tickers = save_batch_to_database(popular_tickers, batch_data, session_id)
        if saved_tickers:
            st.success(f"Saved batch session {session_id} with {len(saved_tickers)} tickers: {', '.join(saved_tickers)}")
            session_data, df = load_batch_session(session_id)
            if session_data and not df.empty:
                st.info(f"Verified batch session {session_id}: {len(df)} records for tickers {df['ticker'].unique().tolist()}")
            else:
                st.warning(f"Batch session {session_id} saved but no data retrieved. Check database.")
        else:
            st.error(f"Failed to save batch session {session_id}: No tickers saved")
    else:
        st.warning("No valid data to save for batch analysis")
    st.session_state['batch_tickers'] = []
    st.session_state['ticker'] = ''  # Clear ticker to prevent misuse
    logger.info(f"Completed batch analysis for session_id {session_id}, session_state.ticker={st.session_state.get('ticker', 'None')}")

st.markdown("---")
st.caption("2025 Stock Sentiment & Trading AI | Powered by Yahoo Finance + NewsAPI + Gemini")