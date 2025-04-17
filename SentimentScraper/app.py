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

# --- Load environment variables ---
load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit page config ---
st.set_page_config(page_title="Stock News Sentiment", page_icon="📈", layout="wide")
st.title("📊 Stock News Sentiment Analysis")

st.markdown("This app analyzes Yahoo Finance news for a stock and calculates sentiment using Google Gemini.")

# --- Session state setup ---
if "sentiment_df" not in st.session_state:
    st.session_state.sentiment_df = None

# --- Parse relative time from Yahoo ---
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

# --- Fetch news from Yahoo Finance ---
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
                parts = publishing_div.text.strip().split("•")
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

# --- Fetch full article text ---
def fetch_full_article_text(url: str) -> Optional[str]:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logger.error(f"Failed to fetch article from {url}: {e}")
        return None

# --- Gemini: Summarize article ---
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

# --- Gemini: Sentiment analysis ---
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

# --- Sentiment calculation ---
def calculate_average_sentiment(scores):
    valid_scores = [s for s in scores if isinstance(s, (int, float))]
    return sum(valid_scores) / len(valid_scores) if valid_scores else 0

# --- Analyze all articles ---
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

# --- User Input ---
ticker_input = st.text_input("Enter Stock Ticker Symbol:", placeholder="e.g., AAPL")

if ticker_input:
    ticker = ticker_input.strip().upper()
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Gemini API key not found. Please check your .env file.")
        st.stop()

    with st.spinner(f"Fetching news for {ticker}..."):
        headlines = fetch_yahoo_news(ticker)
        if not headlines:
            st.error("No news found.")
            st.stop()

        df = pd.DataFrame({
            'headline': [h['title'] for h in headlines],
            'url': [h['url'] for h in headlines],
            'source': [h['source']['name'] for h in headlines],
            'publishedAt': [h['publishedAt'] for h in headlines]
        })

        df['publishedAt_dt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
        df = df.sort_values(by='publishedAt_dt', ascending=False).reset_index(drop=True)

    # --- Analyze or Load from State ---
    if st.session_state.sentiment_df is None:
        with st.spinner("Analyzing sentiment..."):
            analyzed_df = analyze_headlines(df, api_key)
            st.session_state.sentiment_df = analyzed_df
    else:
        analyzed_df = st.session_state.sentiment_df

    # --- Display average ---
    avg_sentiment = calculate_average_sentiment(analyzed_df['combined_sentiment'])
    st.metric("📈 Average Sentiment Score", f"{avg_sentiment:.2f}")

    # --- Sentiment Category ---
    analyzed_df['sentiment_category'] = analyzed_df['combined_sentiment'].apply(
        lambda x: "Positive" if x > 0 else "Neutral" if x == 0 else "Negative"
    )

    # --- Charts ---
    col1, col2 = st.columns(2)

    with col1:
        analyzed_df['short_headline'] = analyzed_df['headline'].apply(lambda x: ' '.join(x.split()[:4]) + "...")
        bar_fig = px.bar(
            analyzed_df,
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

    # --- Headlines & summaries ---
    st.subheader("📰 Headlines & Summaries")
    for _, row in analyzed_df.iterrows():
        st.markdown(f"**[{row['headline']}]({row['url']})**")
        st.markdown(f"*{row['summary']}*")
        st.caption(f"Published: {row['publishedAt']} | Sentiment Score: {row['combined_sentiment']}")
        st.divider()

    # --- Download CSV ---
    st.download_button(
        label="📥 Download CSV",
        data=analyzed_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_sentiment.csv",
        mime="text/csv"
    )

    # --- Optional: Re-run analysis ---
    if st.button("🔄 Re-analyze"):
        with st.spinner("Re-analyzing..."):
            analyzed_df = analyze_headlines(df, api_key)
            st.session_state.sentiment_df = analyzed_df
            st.experimental_rerun()

st.markdown("---")
st.caption("© 2025 Stock Sentiment AI | Powered by Yahoo Finance + Gemini Pro")
