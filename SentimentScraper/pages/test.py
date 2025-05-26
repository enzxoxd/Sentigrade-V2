import yfinance as yf
from datetime import date

ticker = yf.Ticker("AAPL")
data = ticker.history(start="2025-05-22", end="2025-05-27", interval="1d")
print(data)
