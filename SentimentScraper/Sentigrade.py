from datetime import datetime, timedelta
import os
import requests
from bs4 import BeautifulSoup
import re
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from typing import Optional, List, Dict
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
import time

# --- Load environment variables ---
load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit page config ---
st.set_page_config(page_title="Sentigrade - Finviz Edition", page_icon="📈", layout="wide")
st.title("📊 Sentigrade - Finviz Edition")
st.caption("Financial sentiment analysis powered by Finviz news")

# --- Configuration ---
TARGET_TICKERS = ['SPY', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META']
HISTORICAL_CSV_PATH = 'historical_sentiment.csv'

# --- Finviz Scraper Functions ---
BASE_URL = "https://finviz.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

@st.cache_data(ttl=3600)
def get_finviz_news(ticker: str, limit: int = 2) -> List[Dict]:
    """Fetch news headlines for a specific ticker from Finviz"""
    url = f"{BASE_URL}/quote.ashx?t={ticker}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        headlines = []
        
        # Look for news in the news table
        news_table = soup.find("table", {"id": "news-table"})
        if news_table:
            for row in news_table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2:
                    time_cell = cells[0].text.strip()
                    news_cell = cells[1]
                    
                    link_tag = news_cell.find("a")
                    if link_tag:
                        title = link_tag.text.strip()
                        link = link_tag.get("href", "")
                        
                        # Make relative URLs absolute
                        if link.startswith("/"):
                            link = BASE_URL + link
                        
                        # Parse time information
                        published_at = parse_finviz_time(time_cell)
                        
                        headlines.append({
                            "title": title,
                            "url": link,
                            "description": title,
                            "publishedAt": published_at.isoformat() if published_at else datetime.now().isoformat(),
                            "source": "Finviz",
                            "origin": "Finviz",
                            "time_text": time_cell
                        })
                        
                        if len(headlines) >= limit:
                            break

        logger.info(f"Found {len(headlines)} headlines for {ticker} from Finviz")
        return headlines

    except Exception as e:
        logger.error(f"Failed to fetch Finviz news for {ticker}: {e}")
        return []

def parse_finviz_time(time_text: str) -> Optional[datetime]:
    """Parse Finviz time format to datetime object"""
    now = datetime.now()
    try:
        time_text = time_text.strip()
        
        # Handle different time formats from Finviz
        if ":" in time_text and len(time_text.split()) == 1:
            # Format: "10:30AM" - today
            try:
                time_obj = datetime.strptime(time_text, "%I:%M%p")
                return now.replace(hour=time_obj.hour, minute=time_obj.minute, second=0, microsecond=0)
            except:
                pass
        
        if re.match(r"\w{3}-\d{2}", time_text):
            # Format: "Dec-15" - this year
            try:
                month_day = datetime.strptime(f"{time_text}-{now.year}", "%b-%d-%Y")
                return month_day
            except:
                pass
        
        if re.match(r"\d{2}-\d{2}", time_text):
            # Format: "12-15" - this year
            try:
                month_day = datetime.strptime(f"{time_text}-{now.year}", "%m-%d-%Y")
                return month_day
            except:
                pass
        
        # Try standard date parsing
        return parse(time_text, default=now)
        
    except Exception as e:
        logger.debug(f"Could not parse time '{time_text}': {e}")
        return now

# --- Helper Functions ---
def fetch_full_article_text(url: str) -> Optional[str]:
    """Fetch full article text using newspaper3k"""
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        article_text = article.text.strip()
        if len(article_text) < 100:
            return None
        
        # Truncate to save on API costs
        if len(article_text) > 2000:
            article_text = article_text[:2000] + "..."
            
        return article_text
    except Exception as e:
        logger.debug(f"Failed to fetch article from {url}: {e}")
        return None

def gemini_generate_summary(article_text: str, api_key: str) -> str:
    """Generate article summary using Gemini"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        if len(article_text) > 1500:
            article_text = article_text[:1500] + "..."
        
        prompt = f"""
        Summarize this financial news in 1-2 sentences focusing on key market impact:
        {article_text}
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini summarization failed: {e}")
        return "Summary unavailable."

def gemini_analyze_sentiment(headline: str, summary: str, api_key: str) -> float:
    """Analyze sentiment using Gemini"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""
        Rate sentiment for stock impact (-10 to 10):
        Headline: {headline}
        Summary: {summary}
        
        Return only the number.
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Extract number from response
        match = re.search(r"-?\d+(?:\.\d+)?", response_text)
        if match:
            score = float(match.group())
            return max(-10.0, min(10.0, score))
        else:
            logger.warning(f"Could not extract sentiment score from: {response_text}")
            return 0.0
            
    except Exception as e:
        logger.error(f"Gemini sentiment analysis failed: {e}")
        return 0.0

def analyze_headlines(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """Analyze headlines for sentiment"""
    if df.empty or not {'title', 'url'}.issubset(df.columns):
        logger.error("Input DataFrame missing required columns or empty")
        return pd.DataFrame()

    df = df.copy()
    df['summary'] = ""
    df['combined_sentiment'] = 0.0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, row in df.iterrows():
        progress = (i + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f"Processing article {i + 1}/{len(df)}: {row['title'][:50]}...")
        
        headline = str(row['title']).strip()
        url = str(row['url']).strip()
        
        if not headline or not url or not url.startswith('http'):
            continue

        try:
            # For cost optimization, try headline-only analysis first
            if len(headline) < 80:
                summary = f"Headline analysis: {headline}"
            else:
                article_text = fetch_full_article_text(url)
                if article_text and len(article_text.strip()) > 100:
                    summary = gemini_generate_summary(article_text, api_key)
                else:
                    summary = f"Headline analysis: {headline}"
                
        except Exception as e:
            logger.debug(f"Using headline for sentiment analysis: {e}")
            summary = f"Headline analysis: {headline}"

        # Analyze sentiment
        sentiment_score = gemini_analyze_sentiment(headline, summary, api_key)
        
        df.at[i, 'summary'] = summary
        df.at[i, 'combined_sentiment'] = sentiment_score
        
        # Rate limiting
        time.sleep(0.3)

    progress_bar.empty()
    status_text.empty()
    
    return df

# --- CSV Integration Functions (FIXED) ---
def load_historical_csv() -> pd.DataFrame:
    """Load existing historical sentiment CSV file with proper error handling"""
    if os.path.exists(HISTORICAL_CSV_PATH):
        try:
            df = pd.read_csv(HISTORICAL_CSV_PATH)
            logger.info(f"Loaded {len(df)} records from {HISTORICAL_CSV_PATH}")
            
            # Validate required columns exist
            required_columns = ['ticker', 'headline', 'publishedAt', 'combined_sentiment', 'source', 'url']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns in CSV: {missing_columns}")
                # Add missing columns with default values
                for col in missing_columns:
                    if col == 'combined_sentiment':
                        df[col] = 0.0
                    else:
                        df[col] = ""
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return pd.DataFrame()
    else:
        logger.info("No existing CSV file found, will create new one")
        return pd.DataFrame()

def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean data before saving"""
    if df.empty:
        return df
    
    # Ensure required columns exist
    required_columns = ['ticker', 'headline', 'publishedAt', 'combined_sentiment', 'source', 'url']
    
    for col in required_columns:
        if col not in df.columns:
            if col == 'combined_sentiment':
                df[col] = 0.0
            else:
                df[col] = ""
    
    # Clean and validate data
    df = df.copy()
    
    # Clean ticker symbols
    df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()
    
    # Ensure headlines are strings and not empty
    df['headline'] = df['headline'].astype(str).str.strip()
    df = df[df['headline'] != '']
    df = df[df['headline'] != 'nan']
    
    # Validate and standardize publishedAt
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    df = df.dropna(subset=['publishedAt'])
    df['publishedAt'] = df['publishedAt'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Ensure sentiment is numeric
    df['combined_sentiment'] = pd.to_numeric(df['combined_sentiment'], errors='coerce').fillna(0.0)
    
    # Ensure URLs are valid strings
    df['url'] = df['url'].astype(str).str.strip()
    df = df[df['url'].str.startswith('http')]
    
    # Ensure source is string
    df['source'] = df['source'].astype(str).str.strip()
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['ticker', 'headline', 'publishedAt'], keep='last')
    
    return df

def save_to_csv(analyzed_df: pd.DataFrame, ticker: str) -> bool:
    """Save/append sentiment data to CSV file with proper validation"""
    try:
        # Load existing data
        existing_df = load_historical_csv()
        
        # Prepare new data in dashboard-compatible format
        new_records = []
        for _, row in analyzed_df.iterrows():
            record = {
                'ticker': str(ticker).strip().upper(),
                'headline': str(row['title']).strip(),
                'publishedAt': str(row['publishedAt']),
                'combined_sentiment': float(row.get('combined_sentiment', 0.0)),
                'source': str(row.get('source', 'Finviz')),
                'url': str(row['url']).strip(),
                'summary': str(row.get('summary', '')),
                'description': str(row.get('description', '')),
                'origin': str(row.get('origin', 'Finviz')),
                'time_text': str(row.get('time_text', '')),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            new_records.append(record)
        
        if not new_records:
            logger.warning("No valid records to save")
            return False
        
        new_df = pd.DataFrame(new_records)
        
        # Validate and clean new data
        new_df = validate_and_clean_data(new_df)
        
        if new_df.empty:
            logger.warning("No valid records after cleaning")
            return False
        
        # Combine with existing data
        if not existing_df.empty:
            # Validate existing data too
            existing_df = validate_and_clean_data(existing_df)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        # Final deduplication
        combined_df = combined_df.drop_duplicates(
            subset=['ticker', 'headline', 'publishedAt'], 
            keep='last'
        )
        
        # Save to CSV with proper encoding
        combined_df.to_csv(HISTORICAL_CSV_PATH, index=False, encoding='utf-8')
        
        logger.info(f"Successfully saved {len(new_df)} new records to {HISTORICAL_CSV_PATH}")
        logger.info(f"Total records in CSV: {len(combined_df)}")
        
        # Verify the save was successful
        if os.path.exists(HISTORICAL_CSV_PATH):
            verify_df = pd.read_csv(HISTORICAL_CSV_PATH)
            if len(verify_df) >= len(new_df):
                return True
            else:
                logger.error("CSV verification failed - file may be corrupted")
                return False
        else:
            logger.error("CSV file not found after save attempt")
            return False
        
    except Exception as e:
        logger.error(f"Failed to save to CSV: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# --- Main Analysis Function ---
def analyze_ticker(ticker: str, is_batch: bool = False) -> Optional[pd.DataFrame]:
    """Main function to analyze ticker sentiment"""
    ticker = ticker.strip().upper()
    if not ticker:
        st.error("No valid ticker symbol provided")
        return None

    # Check for cached results
    cache_key = f"analyzed_data_{ticker}"
    if not is_batch and cache_key in st.session_state:
        if st.button(f"🔄 Refresh {ticker} Analysis", key=f"refresh_{ticker}"):
            if cache_key in st.session_state:
                del st.session_state[cache_key]
            st.rerun()
        else:
            return st.session_state[cache_key]

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")
        return None

    # Fetch news from Finviz
    with st.spinner(f"Fetching Finviz news for {ticker}..."):
        articles = get_finviz_news(ticker, limit=2)
        
        if not articles:
            st.warning(f"No news articles found for {ticker} on Finviz")
            return None

        # Filter valid articles
        valid_articles = [
            article for article in articles
            if article.get('title') and article.get('url')
            and len(str(article['title']).strip()) > 10
            and str(article['url']).startswith("http")
        ]

        if len(valid_articles) < 1:
            st.warning(f"No valid articles found for {ticker}")
            return None

        # Create DataFrame
        df = pd.DataFrame({
            'title': [a['title'] for a in valid_articles],
            'url': [a['url'] for a in valid_articles],
            'source': [a['source'] for a in valid_articles],
            'origin': [a['origin'] for a in valid_articles],
            'publishedAt': [a['publishedAt'] for a in valid_articles],
            'description': [a['description'] for a in valid_articles],
            'time_text': [a.get('time_text', '') for a in valid_articles]
        })

        # Clean and prepare data
        df = df.dropna(subset=['title', 'url'])
        df = df[df['title'].astype(str).str.strip() != '']
        df = df[df['url'].astype(str).str.startswith('http')]
        
        # Parse dates
        df['publishedAt_dt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
        df = df.sort_values(by='publishedAt_dt', ascending=False, na_position='last').reset_index(drop=True)

    # Analyze sentiment
    if not df.empty:
        with st.spinner(f"Analyzing sentiment for {ticker} ({len(df)} articles)..."):
            analyzed_df = analyze_headlines(df.copy(), api_key)
            
            if analyzed_df.empty or 'combined_sentiment' not in analyzed_df.columns:
                st.error(f"Sentiment analysis failed for {ticker}")
                return None

            # Cache results
            if not is_batch:
                st.session_state[cache_key] = analyzed_df

    else:
        st.error(f"No articles to analyze for {ticker}")
        return None

    # Calculate metrics
    sentiment_scores = [s for s in analyzed_df['combined_sentiment'] if pd.notna(s)]
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
    with col2:
        st.metric("Articles Analyzed", len(analyzed_df))
    with col3:
        positive_count = len(analyzed_df[analyzed_df['combined_sentiment'] > 0])
        st.metric("Positive Articles", f"{positive_count}/{len(analyzed_df)}")

    # Categorize sentiment
    analyzed_df['sentiment_category'] = analyzed_df['combined_sentiment'].apply(
        lambda x: "Positive" if x > 1 else "Negative" if x < -1 else "Neutral"
    )

    # CRITICAL: Save to CSV for dashboard integration
    csv_success = save_to_csv(analyzed_df, ticker)
    if csv_success:
        st.success(f"✅ Data saved to {HISTORICAL_CSV_PATH} for dashboard integration")
        
        # Show verification info
        with st.expander("📊 CSV Integration Status"):
            if os.path.exists(HISTORICAL_CSV_PATH):
                verify_df = pd.read_csv(HISTORICAL_CSV_PATH)
                st.info(f"Total records in CSV: {len(verify_df)}")
                st.info(f"Unique tickers: {verify_df['ticker'].nunique() if 'ticker' in verify_df.columns else 'Unknown'}")
                st.info(f"Date range: {verify_df['publishedAt'].min()} to {verify_df['publishedAt'].max()}" if 'publishedAt' in verify_df.columns else "Date range: Unknown")
    else:
        st.error(f"❌ Failed to save data to CSV - dashboard integration may not work")

    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment bar chart
        chart_df = analyzed_df.copy()
        chart_df['short_title'] = chart_df['title'].apply(
            lambda x: ' '.join(str(x).split()[:6]) + "..." if len(str(x).split()) > 6 else str(x)
        )
        
        bar_fig = px.bar(
            chart_df,
            x='short_title',
            y='combined_sentiment',
            color='combined_sentiment',
            color_continuous_scale='RdYlGn',
            title=f'Sentiment Scores - {ticker}',
            range_color=[-10, 10]
        )
        bar_fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
            xaxis_title="Headlines",
            yaxis_title="Sentiment Score"
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    with col2:
        # Sentiment distribution pie chart
        pie_fig = px.pie(
            analyzed_df,
            names='sentiment_category',
            title=f"Sentiment Distribution - {ticker}",
            color='sentiment_category',
            color_discrete_map={
                "Positive": "green", 
                "Neutral": "gray", 
                "Negative": "red"
            }
        )
        pie_fig.update_layout(height=400)
        st.plotly_chart(pie_fig, use_container_width=True)

    # Display articles
    st.subheader(f"📰 Latest News for {ticker}")
    
    for idx, row in analyzed_df.iterrows():
        with st.expander(f"📄 {str(row['title'])[:80]}... (Sentiment: {row['combined_sentiment']:.1f})"):
            st.markdown(f"**[{row['title']}]({row['url']})**")
            
            if row.get('summary') and 'Summary unavailable' not in str(row['summary']):
                st.markdown(f"*{row['summary']}*")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.caption(f"**Source:** {row['source']}")
            with col_b:
                st.caption(f"**Published:** {row.get('time_text', 'Unknown')}")
            with col_c:
                st.caption(f"**Sentiment:** {row['combined_sentiment']:.2f}")

    # Download button
    st.download_button(
        label=f"⬇️ Download {ticker} Analysis",
        data=analyzed_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_finviz_sentiment_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key=f"download_{ticker}"
    )

    return analyzed_df

# --- Streamlit UI ---
def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = ''
    
    # CSV Integration Status
    st.sidebar.markdown("### 📊 Dashboard Integration")
    csv_exists = os.path.exists(HISTORICAL_CSV_PATH)
    if csv_exists:
        try:
            csv_df = pd.read_csv(HISTORICAL_CSV_PATH)
            st.sidebar.success(f"✅ CSV found: {len(csv_df)} records")
            unique_tickers = csv_df['ticker'].nunique() if 'ticker' in csv_df.columns else 0
            st.sidebar.info(f"📈 Unique tickers: {unique_tickers}")
            st.sidebar.info(f"📁 File: `{HISTORICAL_CSV_PATH}`")
        except Exception as e:
            st.sidebar.error(f"❌ CSV exists but corrupted: {e}")
    else:
        st.sidebar.warning("⚠️ No historical CSV found")
        st.sidebar.info("Run analysis to generate CSV for dashboard")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    analysis_type = st.sidebar.radio(
        "Choose Analysis Type:",
        ["Single Ticker", "Batch Analysis", "CSV Status"]
    )
    
    if analysis_type == "Single Ticker":
        st.header("🎯 Single Ticker Analysis")
        
        # Show target tickers
        st.info(f"**Target Tickers (Cost Optimized):** {', '.join(TARGET_TICKERS)}")
        
        # Ticker input
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker_input = st.text_input(
                "Enter ticker symbol:",
                value=st.session_state.current_ticker,
                placeholder="Choose from: SPY, AAPL, MSFT, NVDA, AMZN, META"
            )
        with col2:
            analyze_btn = st.button("🔍 Analyze", type="primary")

        # Target ticker buttons
        st.markdown("**Select a target ticker:**")
        cols = st.columns(len(TARGET_TICKERS))
        
        selected_target = None
        for i, ticker in enumerate(TARGET_TICKERS):
            if cols[i].button(ticker, key=f"target_{ticker}"):
                selected_target = ticker

        if selected_target:
            st.session_state.current_ticker = selected_target
            st.rerun()

        # Analyze ticker
        if analyze_btn and ticker_input.strip():
            ticker = ticker_input.strip().upper()
            
            if ticker not in TARGET_TICKERS:
                st.warning(f"⚠️ {ticker} is not in the target list. For cost optimization, please choose from: {', '.join(TARGET_TICKERS)}")
            else:
                st.session_state.current_ticker = ticker
                st.markdown(f"## Analysis Results for {ticker}")
                analyze_ticker(ticker)
            
        elif st.session_state.current_ticker:
            ticker = st.session_state.current_ticker
            if ticker in TARGET_TICKERS:
                st.markdown(f"## Analysis Results for {ticker}")
                analyze_ticker(ticker)

    elif analysis_type == "Batch Analysis":
        st.header("📊 Batch Analysis")
        st.info("Analyze all target tickers for cost-efficient batch processing")
        
        st.markdown(f"**Will analyze:** {', '.join(TARGET_TICKERS)}")
        
        if st.button("🚀 Analyze All Target Tickers", type="primary"):
            batch_results = []
            for i, ticker in enumerate(TARGET_TICKERS):
                st.markdown(f"### {i+1}. {ticker}")
                
                try:
                    result = analyze_ticker(ticker, is_batch=True)
                    if result is not None:
                        batch_results.append((ticker, result))
                    
                    st.markdown("---")
                    
                except Exception as e:
                    st.error(f"Failed to analyze {ticker}: {e}")
                    continue
            
            # Summary
            if batch_results:
                st.success(f"✅ Completed batch analysis for {len(batch_results)} tickers")
            else:
                st.error("❌ No tickers were successfully analyzed")

    elif analysis_type == "CSV Status":
        st.header("📁 CSV Integration Status")
        
        if os.path.exists(HISTORICAL_CSV_PATH):
            try:
                df = pd.read_csv(HISTORICAL_CSV_PATH)
                st.success(f"✅ Historical CSV found: {len(df)} records")
                
                # Show CSV info
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Records", len(df))
                    st.metric("Unique Tickers", df['ticker'].nunique() if 'ticker' in df.columns else 0)
                    
                with col2:
                    if 'publishedAt' in df.columns:
                        st.metric("Date Range", f"{df['publishedAt'].min()[:10]} to {df['publishedAt'].max()[:10]}")
                    if 'combined_sentiment' in df.columns:
                        avg_sentiment = df['combined_sentiment'].mean()
                        st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
                
                # Show sample data
                st.subheader("Sample Data (First 10 Records)")
                st.dataframe(df.tail(20), use_container_width=True)
                
                # Show column info
                st.subheader("Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.values,
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isnull().sum().values
                })
                st.dataframe(col_info, use_container_width=True)
                
                # Download button
                st.download_button(
                    label="⬇️ Download Complete CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"historical_sentiment_backup_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Failed to read CSV: {e}")
        else:
            st.warning("⚠️ No historical CSV file found")
            st.info("Run some analyses to generate the CSV file")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🚀 <strong>Sentigrade - Finviz Edition (Cost Optimized)</strong> | 
            Powered by Finviz News + Gemini AI | 
            Built with Streamlit</p>
            <p><em>Focused on SPY, AAPL, MSFT, NVDA, AMZN, META for cost-efficient analysis</em></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()