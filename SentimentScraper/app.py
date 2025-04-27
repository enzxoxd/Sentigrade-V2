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

# --- Load environment variables ---
load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit page config ---
st.set_page_config(page_title="Stock News Sentiment Analysis", page_icon="📰", layout="wide")
st.title("📊 Stock News Sentiment Analysis")
st.markdown("Analyze financial news sentiment without price or backtesting data.")

# --- Session state setup ---
for key in ["sentiment_df"]:
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

def analyze_headlines(df, api_key):
    """Analyze sentiment for each headline in the DataFrame."""
    if not {'headline', 'url', 'source'}.issubset(df.columns):
        logger.error(f"Input DataFrame missing required columns: {df.columns}")
        st.error("Input data missing 'headline' or 'url' or 'source' columns.")
        return pd.DataFrame({'headline': [], 'url': [], 'summary': [], 'combined_sentiment': [], 'source': []})

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
        source = row['source']
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
        return pd.DataFrame({'headline': [], 'url': [], 'summary': [], 'combined_sentiment': [], 'source': []})

    if valid_sentiment_scores == 0:
        logger.error("No valid sentiment scores generated. Check Gemini API or article content.")
        st.error("Sentiment analysis failed: No valid sentiment scores generated. Check API key, network, or article content.")
        return pd.DataFrame({'headline': [], 'url': [], 'summary': [], 'combined_sentiment': [], 'source': []})

    return df

# Sidebar for settings
with st.sidebar:
    st.subheader("Settings")
    ticker = st.text_input("Enter stock ticker symbol (e.g. AAPL)", value="AAPL")
    news_limit = st.slider("Number of news articles to fetch", 1, 20, 10)

# Main app logic
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found in .env file. Please set the GEMINI_API_KEY environment variable.")
    gemini_api_key = None
else:
    gemini_api_key = GEMINI_API_KEY

if ticker and gemini_api_key:
    st.write(f"Fetching news articles for **{ticker.upper()}**...")
    yahoo_news = fetch_yahoo_news(ticker, limit=news_limit)
    newsapi_news = fetch_newsapi_headlines(ticker, limit=news_limit//3)

    combined_news = yahoo_news + newsapi_news
    # Remove duplicates by title
    seen_titles = set()
    unique_news = []
    for article in combined_news:
        if article["title"] not in seen_titles:
            unique_news.append(article)
            seen_titles.add(article["title"])

    if not unique_news:
        st.warning("No news articles found for this ticker.")
    else:
        df_news = pd.DataFrame(unique_news)
        df_news.rename(columns={"title": "headline", "source": "source"}, inplace=True)
        # Extract source name
        df_news['source'] = df_news['source'].apply(lambda x: x.get('name', 'Unknown Source') if isinstance(x, dict) else 'Unknown Source')

        # Filter non-English articles
        df_news['language'] = df_news['headline'].apply(lambda x: detect(x) if isinstance(x, str) else 'unknown')
        df_news = df_news[df_news['language'] == 'en'].drop('language', axis=1)

        st.write("Analyzing sentiment of fetched news articles...")
        sentiment_df = analyze_headlines(df_news, gemini_api_key)
        st.session_state.sentiment_df = sentiment_df

        if not sentiment_df.empty:
            # Add article number
            sentiment_df.insert(0, 'Article #', range(1, len(sentiment_df) + 1))

            # Show table of headlines, summaries, sentiment, and source
            st.subheader("News Sentiment Analysis Results")
            st.dataframe(sentiment_df[['Article #', 'headline', 'summary', 'combined_sentiment', 'source']])

            # Define color mapping for sentiment scores
            colors = ['red' if x < 0 else 'green' for x in sentiment_df['combined_sentiment']]

            # Visualization: Sentiment distribution
            fig = px.bar(sentiment_df, x='Article #', y='combined_sentiment',
                         title="Sentiment Scores by Article",
                         labels={"combined_sentiment": "Sentiment Score", "Article #": "Article Number"},
                         color='combined_sentiment',
                         color_continuous_scale=['red', 'green'])  # Set color scale for sentiment
            st.plotly_chart(fig, use_container_width=True)

            # Time series plot if publishedAt available
            if 'publishedAt' in df_news.columns:
                try:
                    sentiment_df['publishedAt'] = pd.to_datetime(df_news['publishedAt'], errors='coerce')
                    sentiment_df = sentiment_df.dropna(subset=['publishedAt'])
                    if not sentiment_df.empty:
                        fig2 = px.scatter(sentiment_df, x='publishedAt', y='combined_sentiment',
                                          title="Sentiment Scores Over Time",
                                          labels={"publishedAt": "Publication Date", "combined_sentiment": "Sentiment Score"},
                                          hover_data=['headline'],
                                          color='combined_sentiment',
                                          color_continuous_scale=['red', 'green'])  # Set color scale for sentiment
                        st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    logger.warning(f"Failed to plot sentiment over time: {e}")

else:
    st.info("Please enter a ticker symbol. Ensure Gemini API key is set in .env file.")
