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

# --- Parse Relative Time ---
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

# --- Fetch News from Yahoo ---
def fetch_yahoo_news(ticker, limit=10):
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

# --- Article Fetcher with Caching ---
@st.cache_data(show_spinner=False)
def fetch_full_article_text(url: str) -> Optional[str]:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logger.error(f"Failed to fetch article from {url}: {e}")
        return None

# --- Gemini Summary Generator ---
def gemini_generate_summary(article_text, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"Summarize this financial news article in 2-3 concise sentences.\n\nArticle:\n{article_text}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini summarization failed: {e}")
        return "Summary unavailable."

# --- Gemini Sentiment Analysis ---
def gemini_analyze_sentiment(headline, summary, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        Analyze the sentiment of this financial news article.
        Headline: {headline}
        Summary: {summary}
        Provide a single sentiment score from -10 (very negative) to 10 (very positive).
        Respond only with the number.
        """
        response = model.generate_content(prompt)
        match = re.search(r"-?\d+(\.\d+)?", response.text.strip())
        return float(match.group()) if match else 0.0
    except Exception as e:
        logger.error(f"Gemini sentiment analysis failed: {e}")
        return 0.0

# --- Sentiment Calculation ---
def calculate_average_sentiment(scores):
    valid_scores = [s for s in scores if isinstance(s, (int, float))]
    return sum(valid_scores) / len(valid_scores) if valid_scores else 0

# --- Process Article ---
def process_article(row, api_key):
    headline = row['headline']
    url = row['url']
    article_text = fetch_full_article_text(url)

    if article_text and len(article_text.strip()) > 100:
        summary = gemini_generate_summary(article_text, api_key)
        source = "Full Article"
    else:
        summary = "Summary unavailable."
        source = "Headline only"

    sentiment_score = gemini_analyze_sentiment(headline, summary, api_key)
    return summary + f" _(Source: {source})_", sentiment_score

# --- Streamlit UI ---
st.set_page_config(page_title="Stock News Sentiment Analysis", page_icon="📈", layout="wide")
st.title("📊 Stock News Sentiment Analysis")

st.markdown("Analyze recent Yahoo Finance news sentiment for any stock ticker symbol.")

ticker_input = st.text_input("Enter Stock Ticker Symbol:", placeholder="e.g., AAPL")

if ticker_input:
    ticker = ticker_input.strip().upper()
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Gemini API key not found. Please check your .env file.")
        st.stop()

    with st.spinner(f"Fetching news headlines for {ticker}..."):
        headlines = fetch_yahoo_news(ticker)
        if not headlines:
            st.error(f"No news headlines found for ticker {ticker}.")
        else:
            df = pd.DataFrame({
                'headline': [h['title'] for h in headlines],
                'url': [h['url'] for h in headlines],
                'source': [h['source']['name'] for h in headlines],
                'publishedAt': [h['publishedAt'] for h in headlines]
            })

            df['publishedAt_dt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
            df = df.sort_values(by='publishedAt_dt', ascending=False).reset_index(drop=True)

            st.subheader(f"🔎 Analyzing {len(df)} Headlines for {ticker}")
            progress_bar = st.progress(0)
            df['summary'] = None
            df['combined_sentiment'] = None

            for i, row in df.iterrows():
                with st.spinner(f"Processing {i+1}/{len(df)}"):
                    summary, sentiment = process_article(row, api_key)
                    df.at[i, 'summary'] = summary
                    df.at[i, 'combined_sentiment'] = sentiment
                    progress_bar.progress((i + 1) / len(df))

            avg_sentiment = calculate_average_sentiment(df['combined_sentiment'])
            st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")

            # --- Sentiment Charts ---
            df['sentiment_category'] = df['combined_sentiment'].apply(
                lambda x: "Positive" if x > 0 else ("Neutral" if x == 0 else "Negative")
            )
            df['short_headline'] = df['headline'].apply(lambda x: ' '.join(x.split()[:3]) + "..." if len(x.split()) > 3 else x)

            st.subheader("📊 Sentiment Visualizations")
            col1, col2 = st.columns(2)

            with col1:
                bar_fig = px.bar(
                    df,
                    x='short_headline',
                    y='combined_sentiment',
                    color='combined_sentiment',
                    color_continuous_scale='RdYlGn',
                    title='Headline Sentiment Scores',
                    labels={'short_headline': 'Headline', 'combined_sentiment': 'Score'}
                )
                bar_fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(bar_fig, use_container_width=True)

            with col2:
                sentiment_counts = df['sentiment_category'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                pie_fig = px.pie(
                    sentiment_counts,
                    values='Count',
                    names='Sentiment',
                    color='Sentiment',
                    color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'},
                    title='Sentiment Split'
                )
                st.plotly_chart(pie_fig, use_container_width=True)

            # --- Article Summaries ---
            st.subheader("📰 Article Summaries")
            for _, row in df.iterrows():
                st.markdown(f"**[{row['headline']}]({row['url']})**")
                st.markdown(f"*{row['summary']}*")
                st.caption(f"Published: {row['publishedAt']} | Sentiment: {row['combined_sentiment']}")

            # --- CSV Export ---
            st.download_button(
                "⬇️ Download Results as CSV",
                df[['headline', 'url', 'publishedAt', 'combined_sentiment', 'summary']].to_csv(index=False),
                file_name=f"{ticker}_sentiment_analysis.csv",
                mime="text/csv"
            )

st.markdown("---")
st.caption("© 2025 Stock Sentiment AI | Powered by Yahoo Finance & Gemini")
