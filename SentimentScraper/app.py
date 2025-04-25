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
import plotly.graph_objects as go
import logging
import google.generativeai as genai
import yfinance as yf
import sqlite3
import ta
import vectorbt as vbt
import numpy as np

# --- Load environment variables ---
load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit page config ---
st.set_page_config(page_title="Stock News & Trading", page_icon="📈", layout="wide")
st.title("📊 Stock News Sentiment & Trading Analysis")
st.markdown("Combine news sentiment with trading strategies.")

# --- Session state setup ---
for key in ["sentiment_df", "stock_price_df", "aligned_df", "backtest_results"]:
    if key not in st.session_state:
        st.session_state[key] = None

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
def fetch_yahoo_news(ticker, limit=10):
    """Enhanced Yahoo Finance news scraper"""
    url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    articles = []

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        selectors = [
            "a.js-content-viewer",
            "a.subtle-link",
            "li.js-stream-content a",
            "div.Ov\\(h\\) a",
            "h3.Mb\\(5px\\)"
        ]

        for selector in selectors:
            elements = soup.select(selector)
            logger.info(f"Selector '{selector}' found {len(elements)} elements")

            if elements:
                for element in elements:
                    if selector.startswith("h3"):
                        parent_link = element.find_parent("a")
                        if parent_link:
                            element = parent_link

                    headline_tag = element.find("h3") or element
                    if not headline_tag or not headline_tag.text.strip():
                        continue

                    desc_tag = element.find("p") or (element.find_next("p") if not element.find("p") else None)

                    title = headline_tag.text.strip()
                    description = desc_tag.text.strip() if desc_tag and desc_tag.text else ""

                    if element.name == "a" and element.has_attr("href"):
                        link = element["href"]
                    else:
                        link_tag = element.find("a")
                        link = link_tag["href"] if link_tag and link_tag.has_attr("href") else ""

                    if not link:
                        continue

                    url_full = link if link.startswith("http") else f"https://finance.yahoo.com{link}"

                    parent_div = element.parent
                    publishing_div = None

                    for parent_level in range(3):
                        if parent_div:
                            publishing_div = parent_div.find("div", class_=lambda c: c and "publishing" in c.lower())
                            if publishing_div:
                                break
                            parent_div = parent_div.parent

                    source, time_str = "Yahoo Finance", "unknown"
                    published_at = None

                    if publishing_div:
                        pub_text = publishing_div.text.strip()
                        for separator in ["•", "·", "-", "|"]:
                            if separator in pub_text:
                                parts = pub_text.split(separator)
                                if len(parts) >= 2:
                                    source = parts[0].strip()
                                    time_str = parts[1].strip()
                                    published_at = parse_relative_time(time_str)
                                    break

                    if published_at and (datetime.now() - published_at).days > 3:
                        continue

                    if any(a["title"] == title for a in articles):
                        continue

                    articles.append({
                        "title": title,
                        "url": url_full,
                        "publishedAt": published_at.isoformat() if published_at else time_str,
                        "source": {"name": source},
                        "description": description,
                        "origin": "Yahoo"
                    })

                    if len(articles) >= limit:
                        break

                if articles:
                    break

        logger.info(f"Successfully fetched {len(articles)} articles for {ticker}")
        return articles

    except Exception as e:
        logger.error(f"Yahoo scraping error: {str(e)}")
        st.error(f"Yahoo scraping error: {str(e)}")
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

import sqlite3
import pandas as pd
from datetime import datetime

def save_to_database(ticker, development_df):
    """Save sentiment and price data to SQLite database."""
    db_path = "stock_sentiment.db"
    
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        
        # Create table if it doesn’t exist
        c.execute('''CREATE TABLE IF NOT EXISTS sentiment_history
                     (ticker TEXT,
                      date DATE,
                      close_price REAL,
                      avg_sentiment REAL,
                      article_count INTEGER,
                      PRIMARY KEY (ticker, date))''')
        
        # Insert data, ignoring duplicates
        for date, row in development_df.iterrows():
            close_price = row['Close']
            avg_sentiment = row['avg_sentiment'] if not pd.isna(row['avg_sentiment']) else None
            article_count = int(row['article_count']) if not pd.isna(row['article_count']) else 0
            c.execute('''INSERT OR IGNORE INTO sentiment_history
                         (ticker, date, close_price, avg_sentiment, article_count)
                         VALUES (?, ?, ?, ?, ?)''',
                      (ticker, date.strftime('%Y-%m-%d'), close_price, avg_sentiment, article_count))
        
        conn.commit()
    logger.info(f"Saved {len(development_df)} records to database for {ticker}")

def load_ticker_history(ticker):
    """Load historical data for a ticker from the database."""
    db_path = "stock_sentiment.db"
    
    with sqlite3.connect(db_path) as conn:
        query = "SELECT date, close_price, avg_sentiment, article_count FROM sentiment_history WHERE ticker = ? ORDER BY date"
        df = pd.read_sql_query(query, conn, params=(ticker,))
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    return df






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

@st.cache_data(ttl=3600)
def fetch_stock_prices(ticker, start_date=None, end_date=None):
    if not start_date:
        start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    start_date_dt = pd.to_datetime(start_date).normalize()
    end_date_dt = pd.to_datetime(end_date).normalize()

    try:
        logger.info(f"Fetching {ticker} price data from Yahoo Finance...")
        stock_data = yf.download(ticker, start=start_date_dt, end=end_date_dt + timedelta(days=1), progress=False)

        if stock_data.empty:
            st.error(f"No stock price data available for {ticker} between {start_date} and {end_date}.")
            return pd.DataFrame()

        return stock_data
    except Exception as e:
        st.error(f"Failed to fetch stock data for {ticker}: {str(e)}")
        return pd.DataFrame()

def calculate_technical_indicators(close_prices):
    """Calculates common technical indicators."""
    try:
        if not isinstance(close_prices, pd.Series):
            st.error(f"Expected Series, got {type(close_prices)}")
            return pd.DataFrame()

        if close_prices.empty:
            st.error("No Close Price Data - Cannot proceed")
            return pd.DataFrame()

        df = pd.DataFrame({'Close': close_prices})

        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['Signal'] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        return df
    except Exception as e:
        logger.error(f"ожалуй:Technical indicator calculation error: {e}")
        st.error(f"Technical indicator calculation error: {e}")
        return pd.DataFrame()

def add_sentiment_signal(df, threshold=2):
    """Generates trading signals based on sentiment scores."""
    df['signal'] = 0  # Neutral
    df.loc[df['avg_sentiment'] > threshold, 'signal'] = 1  # Buy
    df.loc[df['avg_sentiment'] < -threshold, 'signal'] = -1  # Sell
    return df

def align_price_with_sentiment(price_data, sentiment_data):
    if price_data.empty or sentiment_data.empty:
        return pd.DataFrame()

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

def run_backtest(df, fast_window, slow_window, sentiment_threshold):
    """Runs a simple moving average crossover backtest with sentiment signals."""
    if 'Close' not in df.columns or 'avg_sentiment' not in df.columns:
        st.error("Backtest DataFrame missing 'Close' or 'avg_sentiment' columns.")
        return None, None

    if df.empty:
        st.error("No data to run backtest.")
        return None, None

    fast_ma = vbt.MA.run(df['Close'], window=fast_window)
    slow_ma = vbt.MA.run(df['Close'], window=slow_window)
    entries = fast_ma.ma_crossed_above(slow_ma)

    long_entries = entries & (df['avg_sentiment'] > sentiment_threshold)
    exits = fast_ma.ma_crossed_below(slow_ma)

    pf = vbt.Portfolio.from_signals(df['Close'], entries=long_entries, exits=exits, init_cash=10000)

    return pf.stats(), pf.plot()

# --- Streamlit UI ---
ticker_input = st.text_input("Enter Stock Ticker Symbol:", placeholder="e.g., AAPL")

# Sidebar for settings
with st.sidebar:
    st.subheader("Settings")
    start_date = st.date_input("Start date", datetime.now() - timedelta(days=365))
    end_date = st.date_input("End date", datetime.now())

    st.subheader("Backtesting Parameters")
    enable_backtest = st.checkbox("Enable Backtesting")

    if enable_backtest:
        fast_ma_window = st.slider("Fast MA Window", 5, 50, 20)
        slow_ma_window = st.slider("Slow MA Window", 50, 200, 100)
        sentiment_threshold = st.slider("Sentiment Threshold", -5.0, 5.0, 2.0, step=0.5)

    st.subheader("Risk Management")
    initial_capital = st.number_input("Initial Capital", 1000, 1000000, 10000)
    risk_per_trade = st.slider("Risk per Trade (%)", 1, 5, 2)

# --- Main App Logic ---
if ticker_input:
    ticker = ticker_input.strip().upper()

    if "previous_ticker" not in st.session_state:
        st.session_state.previous_ticker = ""

    if ticker != st.session_state.previous_ticker:
        st.session_state.sentiment_df = None
        st.session_state.stock_price_df = None
        st.session_state.aligned_df = None
        st.session_state.backtest_results = None
        st.session_state.previous_ticker = ticker

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Gemini API key not found in .env. Please set GEMINI_API_KEY.")
        st.stop()

    # Fetch data
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
            logger.info(f"Found {len(valid_articles)} valid Yahoo articles for {ticker}")

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

        logger.info(f"Articles to process: {len(articles_to_process)}")
        for article in articles_to_process:
            logger.info(f"Article: title={article.get('title')}, url={article.get('url')}, publishedAt={article.get('publishedAt')}")

        df = pd.DataFrame({
            'headline': [a['title'] for a in articles_to_process],
            'url': [a['url'] for a in articles_to_process],
            'source': [a['source']['name'] for a in articles_to_process],
            'origin': [a.get('origin', '') for a in articles_to_process],
            'publishedAt': [a['publishedAt'] for a in articles_to_process],
            'description': [a['description'] for a in articles_to_process]
        })

        # Validate DataFrame
        df = df.dropna(subset=['headline', 'url'])
        df = df[df['headline'].str.strip() != '']
        df = df[df['url'].str.startswith('http')]
        logger.info(f"DataFrame after validation: shape={df.shape}, columns={df.columns}")
        logger.info(f"DataFrame head after validation:\n{df.head().to_string()}")

        if df.empty:
            st.error("No valid articles after validation. Please try a different ticker or check data sources.")
            st.stop()

        df['publishedAt_dt'] = pd.to_datetime(df['publishedAt'], errors='coerce', utc=True)
        df = df.sort_values(by='publishedAt_dt', ascending=False).reset_index(drop=True)

    if st.session_state.sentiment_df is None:
        with st.spinner("Analyzing sentiment..."):
            analyzed_df = analyze_headlines(df.copy(), api_key)
            if 'combined_sentiment' not in analyzed_df.columns:
                logger.error("Sentiment analysis failed: combined_sentiment column missing.")
                st.error("Sentiment analysis failed: combined_sentiment column missing. Check API key or input data.")
                st.stop()
            if analyzed_df.empty or analyzed_df['combined_sentiment'].isna().all():
                logger.error("Sentiment analysis produced no valid scores. DataFrame empty or all values are NaN.")
                st.error("Sentiment analysis failed: No valid sentiment scores generated. Check API key, network, or input data.")
                st.stop()
            logger.info(f"Final analyzed DataFrame shape: {analyzed_df.shape}, columns: {analyzed_df.columns}")
            logger.info(f"Final analyzed DataFrame head:\n{analyzed_df.head().to_string()}")
            logger.info(f"Final combined sentiment values: {analyzed_df['combined_sentiment'].tolist()}")
            st.session_state.sentiment_df = analyzed_df
    else:
        analyzed_df = st.session_state.sentiment_df
        if 'combined_sentiment' not in analyzed_df.columns:
            logger.error("Session state sentiment_df missing combined_sentiment column.")
            st.error("Cached sentiment data is invalid. Re-analyzing...")
            st.session_state.sentiment_df = None
            analyzed_df = analyze_headlines(df.copy(), api_key)
            if 'combined_sentiment' not in analyzed_df.columns:
                logger.error("Sentiment analysis failed after retry: combined_sentiment column missing.")
                st.error("Sentiment analysis failed: combined_sentiment column missing. Check API key or input data.")
                st.stop()
            if analyzed_df.empty or analyzed_df['combined_sentiment'].isna().all():
                logger.error("Sentiment analysis produced no valid scores after retry. DataFrame empty or all values are NaN.")
                st.error("Sentiment analysis failed: No valid sentiment scores generated. Check API key, network, or input data.")
                st.stop()
            logger.info(f"Final analyzed DataFrame shape after retry: {analyzed_df.shape}, columns: {analyzed_df.columns}")
            logger.info(f"Final analyzed DataFrame head after retry:\n{analyzed_df.head().to_string()}")
            logger.info(f"Final combined sentiment values after retry: {analyzed_df['combined_sentiment'].tolist()}")
            st.session_state.sentiment_df = analyzed_df

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
    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    daily_sentiment.set_index('date', inplace=True)

    price_df = fetch_stock_prices(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    st.session_state.stock_price_df = price_df

    if not price_df.empty:
        close_col = next((col for col in price_df.columns if 'Close' in col), None)
        if close_col:
            tech_indicators_df = calculate_technical_indicators(price_df[close_col])
            st.session_state.stock_price_df = tech_indicators_df
        else:
            st.error("No 'Close' column found in stock price data.")
            st.session_state.stock_price_df = pd.DataFrame()

        aligned_df = align_price_with_sentiment(price_df, daily_sentiment)

        # Save daily snapshot to database
        save_to_database(ticker, aligned_df)

        # Load historical data from database
        rolling_df = load_ticker_history(ticker)

        # Store aligned_df in session state
        aligned_df.index = pd.to_datetime(aligned_df.index).normalize()
        st.session_state.aligned_df = aligned_df

        # Historical trend visualization
        st.subheader("📈 Rolling Sentiment & Price Over Time")
        if not rolling_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling_df.index,
                y=rolling_df['close_price'],
                name='Close Price',
                yaxis='y1',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=rolling_df.index,
                y=rolling_df['avg_sentiment'],
                name='Avg Sentiment',
                yaxis='y2',
                line=dict(color='orange')
            ))

            fig.update_layout(
                title=f"Rolling Snapshot - {ticker}",
                xaxis=dict(title="Date", tickformat="%b %d", tickangle=-45),
                yaxis=dict(title="Close Price", side='left'),
                yaxis2=dict(title="Avg Sentiment", overlaying='y', side='right'),
                legend=dict(x=0, y=1.1, orientation='h')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No snapshot data available yet. Run the app daily to start building history.")

    # Sentiment Visualizations
    col1, col2 = st.columns(2)
    with col1:
        bar_fig = px.bar(
            analyzed_df.assign(short_headline=analyzed_df['headline'].apply(lambda x: ' '.join(x.split()[:4]) + "...")),
            x='short_headline',
            y='combined_sentiment',
            color='combined_sentiment',
            color_continuous_scale='RdYlGn',
            title='Headline Sentiment Scores',
            labels={'short_headline': 'Headline', 'combined_sentiment': 'Sentiment'}
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

    # Headlines and Summaries
    st.subheader("📰 Headlines & Summaries")
    for _, row in analyzed_df.iterrows():
        st.markdown(f"**[{row['headline']}]({row['url']})**")
        st.markdown(f"*{row['summary']}*")
        st.caption(f"Source: {row['source']} ({row['origin']}) | Published: {row['publishedAt']} | Sentiment Score: {row['combined_sentiment']}")
        st.divider()

    # Risk Management
    initial_entry_value = 0.0
    if not price_df.empty:
        close_col = next((col for col in price_df.columns if 'Close' in col), None)
        if close_col:
            try:
                last_price = price_df[close_col].iloc[-1]
                if isinstance(last_price, (int, float, np.number)):
                    initial_entry_value = float(last_price)
            except Exception as e:
                st.error(f"Error getting last price: {e}")

    st.subheader("Risk Management")
    entry_price = st.number_input("Entry Price", value=initial_entry_value)
    stop_loss_pct = st.slider("Stop Loss (%)", 1, 10, 5)
    stop_loss = entry_price * (1 - (stop_loss_pct / 100))
    risk_amount = initial_capital * (risk_per_trade / 100)
    position_size = risk_amount / abs(entry_price - stop_loss) if entry_price != stop_loss else 0

    st.write(f"Stop Loss: {stop_loss:.2f}")
    st.write(f"Position Size: {position_size:.2f} shares")

    # Download Button
    st.download_button(
        label="⬇️ Download CSV",
        data=analyzed_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_sentiment.csv",
        mime="text/csv"
    )

    # Re-analyze Button
    if st.button("🔁 Re-analyze"):
        with st.spinner("Re-analyzing..."):
            st.session_state.sentiment_df = None
            analyzed_df = analyze_headlines(df.copy(), api_key)
            st.session_state.sentiment_df = analyzed_df
            st.experimental_rerun()

st.markdown("---")
st.caption("2025 Stock Sentiment & Trading AI | Powered by Yahoo Finance + NewsAPI + Gemini")