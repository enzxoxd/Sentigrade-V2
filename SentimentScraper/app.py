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

# --- Load environment variables ---
load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit page config ---
st.set_page_config(page_title="Stock News Sentiment", page_icon="📈", layout="wide")
st.title("📊 Stock News Sentiment Analysis")

st.markdown("Analyze Yahoo Finance news sentiment for any stock using Google Gemini.")

# --- Session state setup ---
for key in ["sentiment_df", "stock_price_df", "aligned_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Helper Functions ---
def parse_relative_time(time_text):
    now = datetime.now()
    try:
        if "ago" in time_text.lower():
            num, unit = re.findall(r"(\d+)\s+(\w+)", time_text)[0]
            num = int(num)
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
    except Exception:
        pass
    return None

def fetch_yahoo_news(ticker, limit=3):
    url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    articles = []

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        anchors = soup.select("a.subtle-link")

        for anchor in anchors:
            headline_tag = anchor.find("h3")
            desc_tag = anchor.find("p")
            parent = anchor.parent

            if not headline_tag:
                continue

            title = headline_tag.text.strip()
            description = desc_tag.text.strip() if desc_tag else ""
            link = anchor["href"]
            url_full = link if link.startswith("http") else f"https://finance.yahoo.com{link}"

            publishing_div = parent.find("div", class_="publishing") if parent else None
            source, time_str = "Yahoo Finance", "unknown"
            published_at = None

            if publishing_div:
                parts = publishing_div.text.strip().split("\u2022")
                if len(parts) == 2:
                    source = parts[0].strip()
                    time_str = parts[1].strip()
                    published_at = parse_relative_time(time_str)

            articles.append({
                "title": title,
                "url": url_full,
                "publishedAt": published_at.isoformat() if published_at else time_str,
                "source": {"name": source},
                "description": description
            })

            if len(articles) >= limit:
                break

        return articles

    except Exception as e:
        st.error(f"Yahoo scraping error: {str(e)}")
        return []
def fetch_newsapi_headlines(ticker, limit=3):
    newsapi_key = os.getenv("NEWSAPI_KEY")
    if not newsapi_key:
        st.warning("NEWSAPI_KEY not found. Skipping NewsAPI.")
        return []

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&"
        f"pageSize={limit}&"
        f"sortBy=publishedAt&"
        f"language=en&"
        f"apiKey={newsapi_key}"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()

        st.write("✅ NewsAPI Raw Response:")
        st.json(news_data)  # See exactly what NewsAPI returns

        if news_data.get("status") != "ok":
            st.warning(f"NewsAPI returned an error: {news_data.get('message', 'Unknown error')}")
            return []

        articles = news_data.get("articles", [])
        st.write(f"🧪 Fetched {len(articles)} NewsAPI articles")

        parsed_articles = []
        for a in articles:
            parsed_articles.append({
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "publishedAt": a.get("publishedAt", ""),
                "source": {"name": a.get("source", {}).get("name", "NewsAPI")},
                "description": a.get("description", "")
            })

        return parsed_articles

    except Exception as e:
        st.warning(f"NewsAPI error: {str(e)}")
        return []


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
        logger.error(f"Gemini sentiment analysis failed: {e}")
        return 0.0

def calculate_average_sentiment(scores):
    valid_scores = [s for s in scores if isinstance(s, (int, float))]
    return sum(valid_scores) / len(valid_scores) if valid_scores else 0

def analyze_headlines(df, api_key):
    df['summary'] = None
    df['combined_sentiment'] = None
    for i, row in df.iterrows():
        headline = row['headline']
        url = row['url']
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
    return df

def fetch_stock_prices(ticker, start_date=None, end_date=None):
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    start_date_dt = pd.to_datetime(start_date).normalize()
    end_date_dt = pd.to_datetime(end_date).normalize()

    try:
        st.info(f"Fetching {ticker} price data from Yahoo Finance...")
        stock_data = yf.download(ticker, start=start_date_dt, end=end_date_dt)
        return stock_data
    except Exception as e:
        st.error(f"Failed to fetch stock data: {str(e)}")
        return pd.DataFrame()

def align_price_with_sentiment(price_data, sentiment_data):
    if price_data.empty:
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

    return merged_df
# --- Main App Logic ---
ticker_input = st.text_input("Enter Stock Ticker Symbol:", placeholder="e.g., AAPL")

if ticker_input:
    ticker = ticker_input.strip().upper()
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Gemini API key not found. Please check your .env file.")
        st.stop()

    with st.spinner(f"Fetching news for {ticker}..."):
        yahoo_headlines = fetch_yahoo_news(ticker, limit=3)
        newsapi_headlines = fetch_newsapi_headlines(ticker, limit=3)

        all_articles = yahoo_headlines + newsapi_headlines

        if not all_articles:
            st.error("No news found.")
            st.stop()

        df = pd.DataFrame({
            'headline': [a['title'] for a in all_articles],
            'url': [a['url'] for a in all_articles],
            'source': [a['source']['name'] for a in all_articles],
            'publishedAt': [a['publishedAt'] for a in all_articles],
            'description': [a['description'] for a in all_articles]
        })

        # Convert to datetime safely
        df['publishedAt_dt'] = pd.to_datetime(df['publishedAt'], errors='coerce', utc=True)
        df = df.sort_values(by='publishedAt_dt', ascending=False).reset_index(drop=True)

    if st.session_state.sentiment_df is None:
        with st.spinner("Analyzing sentiment..."):
            analyzed_df = analyze_headlines(df.copy(), api_key)
            st.session_state.sentiment_df = analyzed_df
    else:
        analyzed_df = st.session_state.sentiment_df

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

    price_df = fetch_stock_prices(ticker)
    st.session_state.stock_price_df = price_df

    aligned_df = align_price_with_sentiment(price_df, daily_sentiment)
    aligned_df.index = pd.to_datetime(aligned_df.index).normalize()
    st.session_state.aligned_df = aligned_df

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

    st.subheader("📈 Price vs. Sentiment Analysis")
    if not aligned_df.empty:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=aligned_df.index,
            y=aligned_df['Close'],
            name='Close Price',
            yaxis='y1',
            mode='lines+markers',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=aligned_df.index,
            y=aligned_df['avg_sentiment'],
            name='Avg Sentiment',
            yaxis='y2',
            mode='lines+markers',
            marker=dict(color='orange')
        ))

        fig.update_layout(
            title=f"{ticker} - Price vs Sentiment",
            xaxis=dict(title="Date", tickformat="%b %d", tickangle=-45),
            yaxis=dict(title="Close Price", side='left'),
            yaxis2=dict(title="Avg Sentiment", overlaying='y', side='right'),
            legend=dict(x=0, y=1.1, orientation='h')
        )

        st.plotly_chart(fig, use_container_width=True)

        if len(aligned_df) > 1 and not aligned_df[['Close', 'avg_sentiment']].isnull().any().any():
            correlation = aligned_df['Close'].corr(aligned_df['avg_sentiment'])
            st.metric("📉 Price-Sentiment Correlation", f"{correlation:.2f}")
        else:
            st.warning("Not enough valid data to compute correlation.")

    st.subheader("📰 Headlines & Summaries")
    for _, row in analyzed_df.iterrows():
        st.markdown(f"**[{row['headline']}]({row['url']})**")
        st.markdown(f"*{row['summary']}*")
        st.caption(f"Source: {row['source']} | Published: {row['publishedAt']} | Sentiment Score: {row['combined_sentiment']}")
        st.divider()

    st.download_button(
        label="⬇️ Download CSV",
        data=analyzed_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_sentiment.csv",
        mime="text/csv"
    )

    if st.button("🔁 Re-analyze"):
        with st.spinner("Re-analyzing..."):
            analyzed_df = analyze_headlines(df.copy(), api_key)
            st.session_state.sentiment_df = analyzed_df
            st.experimental_rerun()

st.markdown("---")
st.caption("2025 Stock Sentiment AI | Powered by Yahoo Finance + NewsAPI + Gemini")
