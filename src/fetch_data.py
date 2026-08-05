"""
fetch_data.py
Pulls SPY daily prices and macro predictors from FRED, for regression
practice. Predictors are configurable via a dict of FRED series codes.
"""

import os
import pandas as pd
import yfinance as yf
from pathlib import Path
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FRED_API_KEY = os.getenv("FRED_API_KEY")

START_DATE = "2010-01-01"
END_DATE = "2026-08-05"

TICKER = "SPY"

# Default predictor set — pass a different dict to fetch_macro_predictors
# to use a different set of series without touching this default
DEFAULT_PREDICTORS = {
    "VIX": "VIXCLS",
    "Yield_Slope": "T10Y2Y",
}


def fetch_spy_prices(start=START_DATE, end=END_DATE, ticker=TICKER):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Close"]]


def fetch_macro_predictors(predictors=None, start=START_DATE, end=END_DATE):
    """
    Pulls an arbitrary set of macro series from FRED.

    predictors: dict of {column_name: fred_series_code}, e.g.
        {"VIX": "VIXCLS", "Yield_Slope": "T10Y2Y"}
        Defaults to DEFAULT_PREDICTORS if not provided.
    """
    if FRED_API_KEY is None:
        raise ValueError(
            "FRED_API_KEY not found. Create a .env file in the project "
            "root with: FRED_API_KEY=your_key_here"
        )

    predictors = predictors or DEFAULT_PREDICTORS
    fred = Fred(api_key=FRED_API_KEY)

    data = {}
    for column_name, series_code in predictors.items():
        print(f"Fetching {column_name} ({series_code})...")
        try:
            data[column_name] = fred.get_series(series_code, observation_start=start, observation_end=end)
        except Exception as e:
            print(f"  Failed to fetch {series_code}: {e}")

    macro_df = pd.DataFrame(data)
    return macro_df


def save_data(spy_prices, macro_df):
    DATA_DIR.mkdir(exist_ok=True)
    spy_prices.to_csv(DATA_DIR / "spy_prices.csv")
    macro_df.to_csv(DATA_DIR / "macro_predictors.csv")
    print(f"Saved SPY prices and macro predictors to {DATA_DIR}/")


def main():
    spy_prices = fetch_spy_prices()
    macro_df = fetch_macro_predictors()
    save_data(spy_prices, macro_df)
    print(spy_prices.tail())
    print(macro_df.tail())


if __name__ == "__main__":
    main()

import io
import time
import requests


def get_sp500_universe():
    """
    Scrapes the current S&P 500 constituent list from Wikipedia,
    including GICS sector classification.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research script)"}

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    tables = pd.read_html(io.StringIO(resp.text))
    sp500_table = tables[0]

    universe = sp500_table[["Symbol", "GICS Sector"]].copy()
    universe.columns = ["Symbol", "Sector"]
    universe["Symbol"] = universe["Symbol"].str.replace(".", "-", regex=False)
    return universe


def fetch_sector_etf_prices(sector_etfs, start=START_DATE, end=END_DATE):
    """
    Fetches prices for a set of sector-representative tickers (e.g.
    sector ETFs like XLK, XLF, or a representative single stock per
    sector). Returns a dict of {sector_name: price_df}.
    """
    import yfinance as yf

    sector_prices = {}
    for sector, ticker in sector_etfs.items():
        print(f"Fetching {sector} ({ticker})...")
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        sector_prices[sector] = df[["Close"]]
        time.sleep(1)

    return sector_prices