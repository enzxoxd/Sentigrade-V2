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

# --- Fetch Stock Closing Price (Closest Previous Trading Day) ---
def fetch_stock_data(ticker, target_date):
    try:
        # Ensure target_date is datetime.date (not datetime.datetime)
        if isinstance(target_date, datetime):
            target_date = target_date.date()
        
        start = target_date - timedelta(days=7)  # fetch a wider range to find closest previous trading day
        end = target_date + timedelta(days=2)    # a bit after target_date as buffer
        data = yf.download(ticker, start=start, end=end, progress=False, interval='1d', actions=False)

        if not data.empty:
            data = data.reset_index()
            data['Date'] = pd.to_datetime(data['Date']).dt.date

            # Filter trading days on or before target_date
            valid_dates = data['Date'][data['Date'] <= target_date]

            if valid_dates.empty:
                # No previous trading day available, try next trading day after target_date
                valid_dates_after = data['Date'][data['Date'] > target_date]
                if valid_dates_after.empty:
                    return pd.DataFrame()
                closest_date = valid_dates_after.min()
            else:
                closest_date = valid_dates.max()

            close_price = data.loc[data['Date'] == closest_date, 'Close'].values[0]

            return pd.DataFrame({
                'ticker': [ticker],
                'published_date': [target_date],
                'closing_price': [close_price]
            })
    except Exception as e:
        print(f"Error fetching data for {ticker} on {target_date}: {e}")

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

        # Remove duplicates for same ticker and headline and publishedAt
        historical_df = historical_df.drop_duplicates(subset=['ticker', 'headline', 'publishedAt'])

        # Remove rows with missing critical info
        historical_df = historical_df.dropna(subset=['ticker', 'publishedAt', 'headline', 'combined_sentiment'])

        options = historical_df['ticker'].dropna().unique().tolist()
    else:
        options = []

    ticker_filter = st.sidebar.multiselect("Select Ticker(s)", options=options, default=options)
    date_from = st.sidebar.date_input("From Date", value=datetime.strptime("2025-05-19", "%Y-%m-%d").date())
    date_to = st.sidebar.date_input("To Date", value=datetime.now())

    if date_from > date_to:
        st.sidebar.error("Invalid date range: 'From Date' must be before 'To Date'.")
        return

    filtered_data = filter_sentiment_data(historical_df, ticker_filter, date_from, date_to)

    if not filtered_data.empty:
        filtered_data = add_closing_prices(filtered_data)
        filtered_data = filtered_data.dropna(subset=['ticker', 'publishedAt', 'headline', 'combined_sentiment', 'closing_price'])

        if filtered_data.empty:
            st.warning("No data available with valid closing prices for the selected filters.")
            return

        filtered_data['publishedAt'] = pd.to_datetime(filtered_data['publishedAt']).dt.date
        filtered_data['combined_sentiment'] = filtered_data['combined_sentiment'].astype(float)
        filtered_data['closing_price'] = filtered_data['closing_price'].astype(float)

        # Rebase prices to 100 from a chosen base date (adjust date as needed)
        base_date = datetime.strptime("2025-05-19", "%Y-%m-%d").date()
        rebased_prices = []

        for ticker in filtered_data['ticker'].unique():
            ticker_data = filtered_data[filtered_data['ticker'] == ticker].copy()
            
            # Find base price - use the earliest available date if base_date doesn't exist
            base_price_data = ticker_data[ticker_data['publishedAt'] == base_date]['closing_price']
            
            if base_price_data.empty:
                # Use the earliest available date as base
                earliest_date = ticker_data['publishedAt'].min()
                base_price_data = ticker_data[ticker_data['publishedAt'] == earliest_date]['closing_price']
                if not base_price_data.empty:
                    st.info(f"Using {earliest_date} as base date for {ticker} (original base date {base_date} not available)")
            
            if not base_price_data.empty:
                base_price = base_price_data.values[0]
                ticker_data['rebased_price'] = (ticker_data['closing_price'] / base_price) * 100
                rebased_prices.append(ticker_data)
            else:
                st.warning(f"Could not find base price for {ticker}")

        if rebased_prices:
            filtered_data = pd.concat(rebased_prices, ignore_index=True)
        else:
            st.error("Could not calculate rebased prices for any ticker")
            return

        # Ensure we have valid rebased prices
        filtered_data = filtered_data.dropna(subset=['rebased_price'])
        
        if filtered_data.empty:
            st.warning("No valid rebased price data available.")
            return

        # --- Display Filtered Table ---
        st.subheader("📰 Filtered Sentiment Data with Closing Price")
        st.dataframe(
            filtered_data[['publishedAt', 'ticker', 'headline', 'combined_sentiment', 'closing_price',
                           'rebased_price', 'source', 'url']].sort_values(by='publishedAt', ascending=False),
            use_container_width=True
        )

        # --- 📊 Detailed Chart: Sentiment (Bar) and Rebased Stock Price (Line) ---
        st.subheader("📈 Detailed Chart: Sentiment (Bar) and Rebased Stock Price (Line)")

        # Debug info
        st.write(f"Data points for chart: {len(filtered_data)}")
        st.write(f"Date range: {filtered_data['publishedAt'].min()} to {filtered_data['publishedAt'].max()}")
        st.write(f"Sentiment range: {filtered_data['combined_sentiment'].min()} to {filtered_data['combined_sentiment'].max()}")
        st.write(f"Rebased price range: {filtered_data['rebased_price'].min()} to {filtered_data['rebased_price'].max()}")

        try:
            # Convert date back to datetime for Altair
            chart_data = filtered_data.copy()
            chart_data['publishedAt'] = pd.to_datetime(chart_data['publishedAt'])
            
            # Create more robust scales
            sentiment_min = chart_data['combined_sentiment'].min()
            sentiment_max = chart_data['combined_sentiment'].max()
            sentiment_range = sentiment_max - sentiment_min
            sentiment_padding = max(sentiment_range * 0.1, 1)  # At least 1 unit padding
            
            price_min = chart_data['rebased_price'].min()
            price_max = chart_data['rebased_price'].max()
            price_range = price_max - price_min
            price_padding = max(price_range * 0.05, 1)  # At least 1 unit padding

            sent_scale = alt.Scale(
                domain=[sentiment_min - sentiment_padding, sentiment_max + sentiment_padding]
            )
            price_scale = alt.Scale(
                domain=[price_min - price_padding, price_max + price_padding]
            )

            base = alt.Chart(chart_data).add_selection(
                alt.selection_interval(bind='scales')
            ).encode(
                x=alt.X('publishedAt:T', title='Date'),
                color=alt.Color('ticker:N', legend=alt.Legend(title="Ticker"))
            )

            sentiment_bar = base.mark_bar(opacity=0.6, size=20).encode(
                y=alt.Y('combined_sentiment:Q', 
                       axis=alt.Axis(title='Sentiment Score', titleColor='blue'), 
                       scale=sent_scale),
                tooltip=['publishedAt:T', 'ticker:N', 'combined_sentiment:Q', 'headline:N']
            )

            rebased_price_line = base.mark_line(point=True, strokeWidth=2).encode(
                y=alt.Y('rebased_price:Q', 
                       axis=alt.Axis(title='Rebased Price', titleColor='red'), 
                       scale=price_scale),
                tooltip=['publishedAt:T', 'ticker:N', 'rebased_price:Q']
            )

            chart = alt.layer(sentiment_bar, rebased_price_line).resolve_scale(
                y='independent'
            ).properties(
                width='container',
                height=500,
                title=alt.TitleParams(
                    text="Sentiment (Bars) vs Rebased Stock Price (Lines)",
                    fontSize=16
                )
            )

            st.altair_chart(chart, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating detailed chart: {str(e)}")
            st.write("Chart data sample:")
            st.write(chart_data.head())

        # --- Download CSV Button ---
        st.download_button(
            label="⬇️ Download Filtered Data as CSV",
            data=filtered_data.to_csv(index=False).encode('utf-8'),
            file_name=f"filtered_sentiment_with_prices_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

        # --- Daily Average Summary ---
        st.subheader("📊 Daily Average Sentiment and Price (All Tickers)")

        daily_summary = filtered_data.groupby('publishedAt').agg(
            avg_sentiment=('combined_sentiment', 'mean'),
            avg_rebased_price=('rebased_price', 'mean'),
            data_points=('combined_sentiment', 'count')
        ).reset_index()

        st.dataframe(daily_summary, use_container_width=True)

        if not daily_summary.empty and len(daily_summary) > 0:
            try:
                # Convert date for summary chart
                summary_chart_data = daily_summary.copy()
                summary_chart_data['publishedAt'] = pd.to_datetime(summary_chart_data['publishedAt'])
                
                min_price_avg = summary_chart_data['avg_rebased_price'].min()
                max_price_avg = summary_chart_data['avg_rebased_price'].max()
                avg_range = max_price_avg - min_price_avg
                avg_padding = max(avg_range * 0.05, 0.1)
                
                lower_bound_avg = min_price_avg - avg_padding
                upper_bound_avg = max_price_avg + avg_padding

                avg_chart = alt.Chart(summary_chart_data).add_selection(
                    alt.selection_interval(bind='scales')
                ).encode(x=alt.X('publishedAt:T', title='Date'))

                sentiment_avg_bar = avg_chart.mark_bar(opacity=0.6, color='green').encode(
                    y=alt.Y('avg_sentiment:Q', axis=alt.Axis(title='Average Sentiment', titleColor='green')),
                    tooltip=['publishedAt:T', 'avg_sentiment:Q', 'data_points:Q']
                )

                rebased_avg_line = avg_chart.mark_line(color='red', strokeWidth=3, point=True).encode(
                    y=alt.Y('avg_rebased_price:Q',
                            scale=alt.Scale(domain=[lower_bound_avg, upper_bound_avg]),
                            axis=alt.Axis(title='Average Rebased Price', titleColor='red')),
                    tooltip=['publishedAt:T', 'avg_rebased_price:Q', 'data_points:Q']
                )

                summary_chart = alt.layer(sentiment_avg_bar, rebased_avg_line).resolve_scale(
                    y='independent'
                ).properties(
                    width='container',
                    height=400,
                    title=alt.TitleParams(
                        text="Daily Average Sentiment (Bars) and Rebased Price (Line)",
                        fontSize=16
                    )
                )

                st.altair_chart(summary_chart, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating summary chart: {str(e)}")
        else:
            st.warning("No daily summary data available for charting.")
        # --- Predictive Sentiment Analysis ---
        st.subheader("🤖 Predictive Sentiment Power Analysis")

        # --- Step 1: Compute +1 Day Return ---
        def compute_future_returns(df, days_forward=1):
            df = df.sort_values(['ticker', 'publishedAt'])
            df['publishedAt'] = pd.to_datetime(df['publishedAt'])

            future_prices = []

            for ticker in df['ticker'].unique():
                ticker_df = df[df['ticker'] == ticker].copy()
                ticker_df = ticker_df.sort_values(by='publishedAt')
                ticker_df.set_index('publishedAt', inplace=True)

                ticker_df[f'future_price_{days_forward}d'] = ticker_df['closing_price'].shift(-days_forward)
                ticker_df[f'return_{days_forward}d'] = (
                    (ticker_df[f'future_price_{days_forward}d'] - ticker_df['closing_price']) /
                    ticker_df['closing_price']
                )
                future_prices.append(ticker_df.reset_index())

            return pd.concat(future_prices, ignore_index=True)

        filtered_data = compute_future_returns(filtered_data, days_forward=1)

        # --- Step 2: Label Sentiment and Movement ---
        filtered_data['sentiment_label'] = filtered_data['combined_sentiment'].apply(lambda x: 'pos' if x > 0 else 'neg')
        filtered_data['price_movement'] = filtered_data['return_1d'].apply(lambda x: 'up' if x > 0 else 'down')

        # --- Step 3: Accuracy Check ---
        accuracy = (filtered_data['sentiment_label'] == filtered_data['price_movement']).mean()
        st.metric(label="🧠 Sentiment vs. Price Movement Accuracy (+1 Day)", value=f"{accuracy:.2%}")

        # --- Step 4: Boxplot Visualization (Optional Matplotlib) ---
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            st.subheader("📦 Return Distribution by Sentiment")

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=filtered_data, x='sentiment_label', y='return_1d', ax=ax)
            ax.set_title('Next-Day Return by Sentiment Label')
            st.pyplot(fig)

        except Exception as e:
            st.warning(f"Boxplot error: {e}")

        # --- Step 5: Altair Scatterplot ---
        st.subheader("📉 Sentiment vs. Next-Day Return")

        scatter_data = filtered_data.copy()
        scatter_data['publishedAt'] = pd.to_datetime(scatter_data['publishedAt'])

        scatter_chart = alt.Chart(scatter_data).mark_circle(size=60, opacity=0.5).encode(
            x=alt.X('combined_sentiment:Q', title="Sentiment Score"),
            y=alt.Y('return_1d:Q', title="Next-Day Return"),
            color='ticker:N',
            tooltip=['publishedAt:T', 'headline:N', 'combined_sentiment:Q', 'return_1d:Q']
        ).properties(
            width='container',
            height=400,
            title="Sentiment Score vs. 1-Day Forward Return"
        )

        st.altair_chart(scatter_chart, use_container_width=True)

        # --- Step 6: Simple Logistic Model ---
        st.subheader("🧪 Predictive Model: Logistic Regression")

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix

        model_data = filtered_data.dropna(subset=['combined_sentiment', 'return_1d'])
        model_data['target'] = (model_data['return_1d'] > 0).astype(int)

        X = model_data[['combined_sentiment']]
        y = model_data['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        st.write("**Classification Report**")
        st.dataframe(pd.DataFrame(report).transpose())

        st.write("**Confusion Matrix**")
        st.write(pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['Predicted Down', 'Predicted Up'], index=['Actual Down', 'Actual Up']))

    else:
        st.warning("No results match the selected filters.")

if __name__ == "__main__":
    main()