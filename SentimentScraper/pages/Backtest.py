import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf
import altair as alt

# --- Streamlit App Setup ---
st.set_page_config(page_title="Sentiment Analysis Results", page_icon="📊", layout="wide")
st.title("📊 Sentiment Analysis Results")


# --- Load Historical Sentiment Data ---
def load_historical_data(file_path='historical_sentiment.csv'):
    if os.path.isfile(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()


# --- Clean Ticker Symbol ---
def clean_ticker(ticker):
    return str(ticker).strip().lstrip('$').upper()


# --- Fetch Stock Closing Price ---
def fetch_stock_data(ticker, date):
    try:
        data = yf.download(ticker, start=date, end=date + timedelta(days=1), progress=False, actions=False)
        if not data.empty and 'Close' in data.columns:
            return pd.DataFrame({'ticker': [ticker], 'published_date': [date], 'closing_price': [data['Close'].iloc[0]]})
    except Exception:
        pass
    return pd.DataFrame()


# --- Add Closing Prices to Sentiment Data ---
def add_closing_prices(df):
    df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
    df['published_date'] = df['publishedAt'].dt.date
    df['ticker'] = df['ticker'].apply(clean_ticker)

    closing_prices = []

    for ticker in df['ticker'].dropna().unique():
        sub_df = df[df['ticker'] == ticker]
        for date in sub_df['published_date'].unique():
            price_df = fetch_stock_data(ticker, date)
            if not price_df.empty:
                closing_prices.append(price_df)

    if closing_prices:
        all_prices = pd.concat(closing_prices, ignore_index=True)
        merged = pd.merge(df, all_prices, how='left', on=['ticker', 'published_date'])
        merged = merged.dropna(subset=['closing_price'])
        merged['closing_price'] = merged['closing_price'].astype(float)
        return merged.drop(columns=['published_date'])

    df['closing_price'] = None
    return df


# --- Filter Sentiment Data ---
def filter_sentiment_data(historical_df, ticker_filter, date_from, date_to):
    return historical_df[
        (historical_df['ticker'].isin(ticker_filter)) &
        (historical_df['publishedAt'] >= pd.to_datetime(date_from)) &
        (historical_df['publishedAt'] <= pd.to_datetime(date_to))
    ]


# --- Main App ---
def main():
    historical_df = load_historical_data()

    if not historical_df.empty:
        historical_df['publishedAt'] = pd.to_datetime(historical_df['publishedAt'], errors='coerce')
        historical_df['ticker'] = historical_df['ticker'].apply(clean_ticker)

        # Remove duplicates for same ticker and same headline
        historical_df = historical_df.drop_duplicates(subset=['ticker', 'headline', 'publishedAt'])

        historical_df = historical_df.dropna(subset=['ticker', 'publishedAt', 'headline', 'combined_sentiment'])
        options = historical_df['ticker'].dropna().unique().tolist()
    else:
        options = []

    ticker_filter = st.sidebar.multiselect("Select Ticker(s)", options=options, default=options)
    date_from = st.sidebar.date_input("From Date", value=datetime.now() - timedelta(days=30))
    date_to = st.sidebar.date_input("To Date", value=datetime.now())

    filtered_data = filter_sentiment_data(historical_df, ticker_filter, date_from, date_to)

    if not filtered_data.empty:
        filtered_data = add_closing_prices(filtered_data)
        filtered_data = filtered_data.dropna(subset=['ticker', 'publishedAt', 'headline', 'combined_sentiment', 'closing_price'])

        filtered_data['publishedAt'] = pd.to_datetime(filtered_data['publishedAt']).dt.date
        filtered_data['combined_sentiment'] = filtered_data['combined_sentiment'].astype(float)
        filtered_data['closing_price'] = filtered_data['closing_price'].astype(float)

        # Rebase prices to 100 from 2025-05-19
        base_date = datetime.strptime("2025-05-19", "%Y-%m-%d").date()
        rebased_prices = []

        for ticker in filtered_data['ticker'].unique():
            base_price = filtered_data[
                (filtered_data['ticker'] == ticker) & (filtered_data['publishedAt'] == base_date)
            ]['closing_price']
            if not base_price.empty:
                base = base_price.values[0]
                sub_df = filtered_data[filtered_data['ticker'] == ticker].copy()
                sub_df['rebased_price'] = (sub_df['closing_price'] / base) * 100
                rebased_prices.append(sub_df)

        if rebased_prices:
            filtered_data = pd.concat(rebased_prices, ignore_index=True)
        else:
            filtered_data['rebased_price'] = None

        # --- Display Filtered Table ---
        st.subheader("📰 Filtered Sentiment Data with Closing Price")
        st.dataframe(
            filtered_data[['publishedAt', 'ticker', 'headline', 'combined_sentiment', 'closing_price',
                           'rebased_price', 'source', 'url']].sort_values(by='publishedAt', ascending=False),
            use_container_width=True
        )

        # --- 📊 Detailed Chart with Dynamic Y-Axis ---
        st.subheader("📈 Detailed Chart: Sentiment (Bar) and Rebased Stock Price (Line)")

        # Dynamic Y-axis bounds for detailed chart
        min_sent = filtered_data['combined_sentiment'].min()
        max_sent = filtered_data['combined_sentiment'].max()
        min_price = filtered_data['rebased_price'].min()
        max_price = filtered_data['rebased_price'].max()

        # Adjust the margins to be tighter around the actual values to avoid zooming out too much
        sent_margin = (max_sent - min_sent) * 0.05  # 5% margin
        price_margin = (max_price - min_price) * 0.05  # 5% margin

        sent_scale = alt.Scale(domain=[min_sent - sent_margin, max_sent + sent_margin])
        price_scale = alt.Scale(domain=[min_price - price_margin, max_price + price_margin])

        # Base chart with common color encoding for 'ticker'
        base = alt.Chart(filtered_data).encode(x=alt.X('publishedAt:T', title='Date'), color='ticker:N')

        # Sentiment bar
        sentiment_bar = base.mark_bar(opacity=0.5).encode(
            y=alt.Y('combined_sentiment:Q', axis=alt.Axis(title='Sentiment Score'), scale=sent_scale),
            tooltip=['publishedAt', 'ticker', 'combined_sentiment']
        )

        # Rebased price line
        rebased_price_line = base.mark_line(point=True).encode(
            y=alt.Y('rebased_price:Q', axis=alt.Axis(title='Price (Rebased to 100)'), scale=price_scale),
            tooltip=['publishedAt', 'ticker', 'rebased_price']
        )

        chart = alt.layer(sentiment_bar, rebased_price_line).resolve_scale(y='independent').properties(
            width='container',
            height=400,
            title="Sentiment (Bar) vs Rebased Stock Price (Line)"
        )

        st.altair_chart(chart, use_container_width=True)

        # --- Download CSV Button ---
        st.download_button(
            label="⬇️ Download Filtered Data as CSV",
            data=filtered_data.to_csv(index=False).encode('utf-8'),
            file_name=f"filtered_sentiment_with_prices_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

        # --- 📊 Daily Average Summary ---
        st.subheader("📊 Daily Average Sentiment and Price (All Tickers)")

        daily_summary = filtered_data.groupby('publishedAt').agg(
            avg_sentiment=('combined_sentiment', 'mean'),
            avg_rebased_price=('rebased_price', 'mean')
        ).reset_index()

        st.dataframe(daily_summary, use_container_width=True)

        # Dynamic y-axis bounds for daily chart
        min_price_avg = daily_summary['avg_rebased_price'].min()
        max_price_avg = daily_summary['avg_rebased_price'].max()
        lower_bound_avg = min_price_avg * 0.9995
        upper_bound_avg = max_price_avg * 1.0005

        avg_chart = alt.Chart(daily_summary).encode(x='publishedAt:T')

        sentiment_avg_bar = avg_chart.mark_bar(opacity=0.4, color='green').encode(
            y=alt.Y('avg_sentiment:Q', axis=alt.Axis(title='Average Sentiment'))
        )

        rebased_avg_line = avg_chart.mark_line(color='black').encode(
            y=alt.Y('avg_rebased_price:Q',
                    scale=alt.Scale(domain=[lower_bound_avg, upper_bound_avg]),
                    axis=alt.Axis(title='Average Rebased Price'))
        )

        summary_chart = alt.layer(sentiment_avg_bar, rebased_avg_line).resolve_scale(y='independent').properties(
            width='container',
            height=400,
            title="Daily Average Sentiment (Bar) and Rebased Price (Line)"
        )

        st.altair_chart(summary_chart, use_container_width=True)

    else:
        st.warning("No results match the selected filters.")


if __name__ == "__main__":
    main()
