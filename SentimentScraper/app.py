from datetime import datetime, timedelta
import os
import requests
from bs4 import BeautifulSoup
import re
import streamlit as st
import pandas as pd
from typing import Optional
from newspaper import Article
import plotly.graph_objects as go
import plotly.express as px

# --- Helper to parse relative or absolute time from Yahoo ---
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

# --- Yahoo Finance Scraper ---
def fetch_yahoo_news(ticker, from_date=None, to_date=None, limit=10):
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

            if published_at:
                pub_date = published_at.date().isoformat()
                if from_date and to_date:
                    if not (from_date <= pub_date <= to_date):
                        continue

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

# --- Fetch Yahoo News Only ---
def fetch_news_headlines(ticker, from_date=None, to_date=None):
    yahoo_articles = fetch_yahoo_news(ticker, from_date, to_date)

    def get_date(article):
        try:
            return datetime.fromisoformat(article.get("publishedAt", "").rstrip("Z"))
        except Exception:
            return datetime.min

    yahoo_articles.sort(key=get_date, reverse=True)

    filtered_articles = []
    today_str = datetime.today().date().isoformat()

    for article in yahoo_articles:
        try:
            pub_date = datetime.fromisoformat(article["publishedAt"]).date()
            pub_date_str = pub_date.isoformat()
            if not from_date or not to_date:
                filtered_articles.append(article)
            elif from_date <= pub_date_str <= to_date or pub_date_str == today_str:
                filtered_articles.append(article)
        except Exception:
            continue

    if not filtered_articles:
        st.warning(f"No Yahoo Finance articles found for '{ticker}' between {from_date} and {to_date}. Showing most recent instead.")
        return yahoo_articles[:10]

    return filtered_articles

# --- Dummy Gemini Functions ---
def gemini_generate_summary(text, api_key):
    return text[:250] + "..."

def gemini_analyze_sentiment(prompt, api_key):
    import random
    return round(random.uniform(-10, 10), 2)

def calculate_average_sentiment(scores):
    valid_scores = [s for s in scores if s is not None]
    return round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0

# --- Fetch Full Article ---
def fetch_full_article_text(url: str) -> Optional[str]:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return None

# --- Process Article ---
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
    except Exception:
        summary = "Summary unavailable."
        prompt = f"""
        Analyze the sentiment of this headline:

        {headline}

        Give a score from -10 (very negative) to 10 (very positive). No explanation.
        """
    sentiment_score = gemini_analyze_sentiment(prompt, api_key)
    return summary, sentiment_score

# --- Streamlit App ---
st.set_page_config(page_title="Stock News Sentiment Analysis", page_icon="📈", layout="wide")

st.title("Stock News Sentiment Analysis")
st.markdown("""
This app scrapes news from Yahoo Finance and analyzes sentiment based on headlines and summaries.
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
                api_key = "dummy-key"
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
                st.success(f"Analysis completed! Analyzed {valid_scores_count} headlines.")

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
st.caption("Yahoo Finance News Sentiment Analyzer | Real-time scraping and sentiment analysis")
