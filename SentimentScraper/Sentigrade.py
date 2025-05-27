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
st.set_page_config(page_title="Sentigrade - Finviz Edition", page_icon="📈", layout="wide")
st.title("📊 Sentigrade - Finviz Edition")
st.caption("Financial sentiment analysis powered by Finviz news")

# --- TARGET TICKERS FOR COST OPTIMIZATION ---
TARGET_TICKERS = ['SPY', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META']

# --- Finviz Scraper Functions ---
BASE_URL = "https://finviz.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

@st.cache_data(ttl=3600)
def get_finviz_news(ticker: str, limit: int = 1) -> List[Dict]:  # Reduced from 10 to 3
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
                    # First cell contains time, second contains headline and link
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
                            "description": title,  # Finviz doesn't provide descriptions
                            "publishedAt": published_at.isoformat() if published_at else time_cell,
                            "source": "Finviz",
                            "origin": "Finviz",
                            "time_text": time_cell
                        })
                        
                        if len(headlines) >= limit:
                            break
        
        # Fallback: Look for news links with alternative method
        if not headlines:
            for tag in soup.find_all("a"):
                href = tag.get("href", "")
                if "news" in href.lower() and tag.text.strip():
                    title = tag.text.strip()
                    link = href if href.startswith("http") else BASE_URL + href
                    
                    headlines.append({
                        "title": title,
                        "url": link,
                        "description": title,
                        "publishedAt": datetime.now().isoformat(),
                        "source": "Finviz",
                        "origin": "Finviz",
                        "time_text": "Recent"
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

def get_target_tickers() -> List[str]:
    """Return the predefined target tickers for cost optimization"""
    return TARGET_TICKERS.copy()

# --- Helper Functions ---
def fetch_full_article_text(url: str) -> Optional[str]:
    """Fetch full article text using newspaper3k - with length limits for cost savings"""
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        # Filter out short articles and limit length for cost savings
        article_text = article.text.strip()
        if len(article_text) < 100:
            return None
        
        # Truncate to save on API costs - 2000 chars should be enough for sentiment
        if len(article_text) > 2000:
            article_text = article_text[:2000] + "..."
            
        return article_text
    except Exception as e:
        logger.debug(f"Failed to fetch article from {url}: {e}")
        return None

def gemini_generate_summary(article_text: str, api_key: str) -> str:
    """Generate article summary using Gemini - optimized for cost"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')  # Using flash model for cost efficiency
        
        # Further truncate for summary to save costs
        if len(article_text) > 1500:
            article_text = article_text[:1500] + "..."
        
        # Shorter, more focused prompt to reduce token usage
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
    """Analyze sentiment using Gemini - optimized prompt for cost efficiency"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')  # Using flash model
        
        # Shorter, more direct prompt to save tokens
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
            # Clamp score to valid range
            return max(-10.0, min(10.0, score))
        else:
            logger.warning(f"Could not extract sentiment score from: {response_text}")
            return 0.0
            
    except Exception as e:
        logger.error(f"Gemini sentiment analysis failed for headline '{headline[:50]}...': {e}")
        return 0.0

def calculate_average_sentiment(scores: List[float]) -> float:
    """Calculate average sentiment from scores"""
    valid_scores = [s for s in scores if isinstance(s, (int, float)) and not np.isnan(s)]
    return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

def analyze_headlines(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """Analyze headlines for sentiment - optimized for cost"""
    if df.empty or not {'title', 'url'}.issubset(df.columns):
        logger.error("Input DataFrame missing required columns or empty")
        return pd.DataFrame()

    df = df.copy()
    df['summary'] = None
    df['combined_sentiment'] = 0.0
    
    processed_count = 0
    valid_sentiment_count = 0

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, row in df.iterrows():
        progress = (i + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f"Processing article {i + 1}/{len(df)}: {row['title'][:50]}...")
        
        headline = row['title']
        url = row['url']
        
        if not isinstance(headline, str) or not headline.strip():
            continue
            
        if not isinstance(url, str) or not url.strip() or not url.startswith('http'):
            continue

        try:
            # For cost optimization, try headline-only analysis first for shorter headlines
            if len(headline) < 80:
                # Short headlines - analyze directly without fetching full article
                summary = f"Headline analysis: {headline}"
            else:
                # Longer headlines - try to fetch article but with strict limits
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
        
        processed_count += 1
        if sentiment_score != 0.0:
            valid_sentiment_count += 1
        
        # Reduced rate limiting - but still present to avoid hitting limits
        time.sleep(0.3)  # Reduced from 0.5 to 0.3

    progress_bar.empty()
    status_text.empty()
    
    logger.info(f"Processed {processed_count}/{len(df)} articles, {valid_sentiment_count} with valid sentiment")
    
    if processed_count == 0:
        st.error("No articles could be processed for sentiment analysis")
        return pd.DataFrame()
    
    return df

# --- Database Functions ---
def save_to_database(ticker: str, data_df: pd.DataFrame, session_id: str = None) -> bool:
    """Save sentiment data to database"""
    if not ticker or data_df.empty:
        return False
        
    db_path = "stock_sentiment.db"
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if not exists
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
        
        df_to_save = data_df.copy()
        if df_to_save.index.name is None:
            df_to_save.index.name = 'date'
        df_to_save = df_to_save.reset_index()
        
        df_to_save['record_date'] = datetime.now().strftime("%Y-%m-%d")
        df_to_save['ticker'] = ticker
        df_to_save['session_id'] = session_id or str(uuid4())
        
        # Add signal column if not present
        if 'signal' not in df_to_save.columns:
            df_to_save['signal'] = df_to_save['avg_sentiment'].apply(
                lambda x: 1 if x > 2 else -1 if x < -2 else 0
            )
        
        df_to_save.to_sql('sentiment_history', conn, if_exists='append', index=False)
        conn.commit()
        
        logger.info(f"Saved {len(df_to_save)} records for {ticker}")
        return True
        
    except Exception as e:
        logger.error(f"Database save error for {ticker}: {e}")
        return False
    finally:
        conn.close()

def load_ticker_history(ticker: str) -> pd.DataFrame:
    """Load historical sentiment data for ticker"""
    db_path = "stock_sentiment.db"
    try:
        conn = sqlite3.connect(db_path)
        query = """
        SELECT date, avg_sentiment, article_count, signal, session_id
        FROM sentiment_history
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT 100
        """
        df = pd.read_sql_query(query, conn, params=(ticker,), parse_dates=['date'])
        return df
    except Exception as e:
        logger.error(f"Database load error for {ticker}: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# --- Main Analysis Function ---
def analyze_ticker(ticker: str, is_batch: bool = False) -> Optional[pd.DataFrame]:
    """Main function to analyze ticker sentiment - cost optimized"""
    ticker = ticker.strip().upper()
    if not ticker:
        st.error("No valid ticker symbol provided")
        return None

    # Check for cached results (longer cache for cost savings)
    cache_key = f"analyzed_data_{ticker}"
    if not is_batch and cache_key in st.session_state:
        if st.button(f"🔄 Refresh {ticker} Analysis", key=f"refresh_{ticker}"):
            del st.session_state[cache_key]
            st.rerun()
        else:
            return st.session_state[cache_key]

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")
        return None

    # Fetch news from Finviz (reduced limit for cost savings)
    with st.spinner(f"Fetching Finviz news for {ticker}..."):
        articles = get_finviz_news(ticker, limit=1)  # Reduced from 10 to 3
        
        if not articles:
            st.warning(f"No news articles found for {ticker} on Finviz")
            return None

        # Filter valid articles
        valid_articles = [
            article for article in articles
            if article.get('title') and article.get('url')
            and len(article['title'].strip()) > 10
            and article['url'].startswith("http")
        ]

        if len(valid_articles) < 1:
            st.warning(f"No valid articles found for {ticker}")
            return None

        # Limit articles for processing (cost optimization)
        articles_to_process = valid_articles[:1]  # Reduced from 10 to 3
        
        # Create DataFrame
        df = pd.DataFrame({
            'title': [a['title'] for a in articles_to_process],
            'url': [a['url'] for a in articles_to_process],
            'source': [a['source'] for a in articles_to_process],
            'origin': [a['origin'] for a in articles_to_process],
            'publishedAt': [a['publishedAt'] for a in articles_to_process],
            'description': [a['description'] for a in articles_to_process],
            'time_text': [a.get('time_text', '') for a in articles_to_process]
        })

        # Clean and prepare data
        df = df.dropna(subset=['title', 'url'])
        df = df[df['title'].str.strip() != '']
        df = df[df['url'].str.startswith('http')]
        
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

            # Cache results for longer (cost optimization)
            if not is_batch:
                st.session_state[cache_key] = analyzed_df

    else:
        st.error(f"No articles to analyze for {ticker}")
        return None

    # Calculate metrics
    avg_sentiment = calculate_average_sentiment(analyzed_df['combined_sentiment'])
    
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

    # Create daily aggregation
    sentiment_by_date = analyzed_df.copy()
    sentiment_by_date['date'] = pd.to_datetime(sentiment_by_date['publishedAt_dt']).dt.normalize()
    
    daily_sentiment = sentiment_by_date.groupby('date').agg(
        avg_sentiment=('combined_sentiment', 'mean'),
        article_count=('title', 'count')
    ).reset_index()
    daily_sentiment.set_index('date', inplace=True)

    # Save to database
    if not is_batch and not daily_sentiment.empty:
        save_to_database(ticker, daily_sentiment)

    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment bar chart
        chart_df = analyzed_df.copy()
        chart_df['short_title'] = chart_df['title'].apply(
            lambda x: ' '.join(x.split()[:6]) + "..." if len(x.split()) > 6 else x
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
        with st.expander(f"📄 {row['title'][:80]}... (Sentiment: {row['combined_sentiment']:.1f})"):
            st.markdown(f"**[{row['title']}]({row['url']})**")
            
            if row.get('summary') and 'Summary unavailable' not in row['summary']:
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

    return daily_sentiment

# --- Streamlit UI ---
def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = ''
    
    # API Cost Information
    st.sidebar.markdown("### 💰 Cost Optimization")
    st.sidebar.info(
        """
        **Optimizations Applied:**
        - Limited to 6 target tickers
        - Max 1 articles per ticker
        - Shorter prompts
        - Using Gemini Flash model
        - Longer caching
        """
    )
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    analysis_type = st.sidebar.radio(
        "Choose Analysis Type:",
        ["Single Ticker", "Batch Analysis", "Historical Data"]
    )
    
    if analysis_type == "Single Ticker":
        st.header("🎯 Single Ticker Analysis")
        
        # Show target tickers
        target_tickers = get_target_tickers()
        st.info(f"**Target Tickers (Cost Optimized):** {', '.join(target_tickers)}")
        
        # Ticker input - validate against target tickers
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
        cols = st.columns(len(target_tickers))
        
        selected_target = None
        for i, ticker in enumerate(target_tickers):
            if cols[i].button(ticker, key=f"target_{ticker}"):
                selected_target = ticker

        if selected_target:
            st.session_state.current_ticker = selected_target
            st.rerun()

        # Analyze ticker - with validation
        if analyze_btn and ticker_input.strip():
            ticker = ticker_input.strip().upper()
            
            if ticker not in target_tickers:
                st.warning(f"⚠️ {ticker} is not in the target list. For cost optimization, please choose from: {', '.join(target_tickers)}")
            else:
                st.session_state.current_ticker = ticker
                st.markdown(f"## Analysis Results for {ticker}")
                analyze_ticker(ticker)
            
        elif st.session_state.current_ticker:
            ticker = st.session_state.current_ticker
            if ticker in target_tickers:
                st.markdown(f"## Analysis Results for {ticker}")
                analyze_ticker(ticker)

    elif analysis_type == "Batch Analysis":
        st.header("📊 Batch Analysis")
        st.info("Analyze all target tickers for cost-efficient batch processing")
        
        target_tickers = get_target_tickers()
        st.markdown(f"**Will analyze:** {', '.join(target_tickers)}")
        
        # Estimate cost
        estimated_articles = len(target_tickers) * 3  # 3 articles per ticker
        st.markdown(f"**Estimated API calls:** ~{estimated_articles * 2} (sentiment + summary)")
        
        if st.button("🚀 Analyze All Target Tickers", type="primary"):
            batch_session_id = str(uuid4())
            
            st.markdown(f"**Batch Session ID:** `{batch_session_id}`")
            
            # Process each target ticker
            batch_results = []
            for i, ticker in enumerate(target_tickers):
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
                
                # Overall sentiment summary
                st.subheader("📊 Batch Summary")
                summary_data = []
                for ticker, data in batch_results:
                    if not data.empty:
                        avg_sent = data['avg_sentiment'].mean()
                        article_count = data['article_count'].sum()
                        summary_data.append({
                            'Ticker': ticker,
                            'Avg Sentiment': round(avg_sent, 2),
                            'Total Articles': int(article_count)
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
            else:
                st.error("❌ No tickers were successfully analyzed")

    elif analysis_type == "Historical Data":
        st.header("📈 Historical Sentiment Data")
        
        target_tickers = get_target_tickers()
        ticker_hist = st.selectbox("Select ticker to view history:", target_tickers)
        
        if ticker_hist:
            with st.spinner(f"Loading historical data for {ticker_hist}..."):
                hist_df = load_ticker_history(ticker_hist)
            
            if not hist_df.empty:
                st.subheader(f"Historical Sentiment for {ticker_hist}")
                
                # Time series chart
                fig = px.line(
                    hist_df.sort_values('date'),
                    x='date',
                    y='avg_sentiment',
                    title=f"Sentiment Trend - {ticker_hist}",
                    markers=True
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.subheader("Historical Records")
                st.dataframe(
                    hist_df.sort_values('date', ascending=False),
                    use_container_width=True
                )
                
                # Download historical data
                st.download_button(
                    label=f"⬇️ Download {ticker_hist} History",
                    data=hist_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{ticker_hist}_sentiment_history.csv",
                    mime="text/csv"
                )
            else:
                st.info(f"No historical data found for {ticker_hist}")

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