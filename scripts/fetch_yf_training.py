"""
Fetch training data from Yahoo Finance
=======================================
Pulls financials + company descriptions for ~330 major global public companies
and saves to data/yf_training_data.pkl.

Usage:
    python scripts/fetch_yf_training.py

The app (app.py) will automatically use this file instead of Forbes data
once it exists. Re-run this script to refresh the training data.

Estimated time: 3-6 minutes (concurrent fetching, 10 workers).
"""

import pickle
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Currency conversion — all financials stored in USD billions
# ---------------------------------------------------------------------------

_FX_CACHE: dict[str, float] = {}
_FX_LOCK  = threading.Lock()


def get_usd_rate(currency: str) -> float:
    """Return the USD value of 1 unit of `currency` (e.g. 1 JPY = 0.0067 USD)."""
    if not currency or currency == "USD":
        return 1.0
    with _FX_LOCK:
        if currency in _FX_CACHE:
            return _FX_CACHE[currency]
    try:
        fx   = yf.Ticker(f"{currency}USD=X")
        rate = fx.fast_info.last_price
        if not rate or rate <= 0:
            hist = fx.history(period="1d")
            rate = float(hist["Close"].iloc[-1]) if not hist.empty else 1.0
        rate = float(rate)
    except Exception:
        rate = 1.0
    with _FX_LOCK:
        _FX_CACHE[currency] = rate
    return rate

OUTPUT = Path(__file__).parent.parent / "data" / "yf_training_data.pkl"

# ---------------------------------------------------------------------------
# Curated universe: ~330 major companies across all sectors + global regions
# Chosen to give good coverage for comparable-company matching
# ---------------------------------------------------------------------------

TICKER_UNIVERSE = [
    # ── Tech — Software ──────────────────────────────────────────────────
    "MSFT",  "GOOGL", "META",  "ORCL",  "SAP",   "CRM",   "INTU",  "ADBE",
    "NOW",   "SNOW",  "WDAY",  "TEAM",  "DDOG",  "MDB",   "SHOP",  "HUBS",
    # ── Tech — Hardware / Devices ────────────────────────────────────────
    "AAPL",  "DELL",  "HPQ",   "HPE",   "CSCO",  "NTAP",  "WDC",   "STX",
    "6758.T",                                         # Sony (Japan)
    # ── Semiconductors ───────────────────────────────────────────────────
    "NVDA",  "TSM",   "INTC",  "AMD",   "QCOM",  "AVGO",  "TXN",   "ASML",
    "LRCX",  "KLAC",  "AMAT",  "MU",    "ARM",
    "005930.KS",                                      # Samsung Electronics
    # ── Healthcare / Pharma ──────────────────────────────────────────────
    "JNJ",   "PFE",   "MRK",   "ABBV",  "LLY",   "BMY",   "GILD",  "AMGN",
    "BIIB",  "REGN",  "NVS",   "RHHBY", "AZN",   "GSK",   "SNY",   "NVO",
    # ── Healthcare — Equipment & Services ────────────────────────────────
    "UNH",   "CVS",   "MDT",   "ABT",   "SYK",   "BSX",   "EW",    "ISRG",
    # ── Banking ──────────────────────────────────────────────────────────
    "JPM",   "BAC",   "WFC",   "C",     "GS",    "MS",    "USB",   "PNC",
    "HSBC",  "BNPQY", "ING",   "CS",    "DB",    "SAN",   "BBVA",  "BCS",
    "8306.T",                                         # Mitsubishi UFJ
    "1398.HK",                                        # ICBC
    # ── Diversified Finance / Asset Management ────────────────────────────
    "BRK-B", "BLK",   "SCHW",  "AXP",   "V",     "MA",    "PYPL",  "SQ",
    # ── Insurance ────────────────────────────────────────────────────────
    "MET",   "PRU",   "AIG",   "ALL",   "TRV",   "CB",
    # ── Energy — Oil & Gas ───────────────────────────────────────────────
    "XOM",   "CVX",   "COP",   "SLB",   "OXY",
    "SHEL",  "BP",    "TTE",   "EQNR",  "ENI",
    "2222.SR",                                        # Saudi Aramco
    # ── Energy — Utilities ───────────────────────────────────────────────
    "NEE",   "DUK",   "SO",    "D",     "AEP",   "EXC",   "XEL",
    "ENEL.MI", "IBE.MC",                             # Enel, Iberdrola
    # ── Retail / E-commerce ──────────────────────────────────────────────
    "AMZN",  "WMT",   "TGT",   "COST",  "HD",    "LOW",   "TJX",   "EBAY",
    "BABA",  "JD",    "PDD",   "SE",
    "7203.T",                                         # Toyota (also retail proxy)
    # ── Consumer — Food & Beverage ───────────────────────────────────────
    "KO",    "PEP",   "MCD",   "SBUX",  "YUM",   "CMG",
    "NESN.SW", "UL",  "DANOY", "DEO",
    # ── Consumer Goods / Personal Products ───────────────────────────────
    "PG",    "CL",    "KMB",   "EL",    "ULVR.L",
    # ── Telecom ──────────────────────────────────────────────────────────
    "T",     "VZ",    "TMUS",
    "VOD",   "DTE.DE", "ORAN", "TEF",   "NPPXF",
    "9984.T",                                         # SoftBank
    # ── Media / Entertainment ────────────────────────────────────────────
    "DIS",   "NFLX",  "PARA",  "WBD",   "SPOT",
    # ── Aerospace & Defence ──────────────────────────────────────────────
    "BA",    "LMT",   "RTX",   "NOC",   "GD",    "HII",
    "AIR.PA",                                         # Airbus
    # ── Capital Goods / Machinery ────────────────────────────────────────
    "CAT",   "DE",    "HON",   "GE",    "EMR",   "ROK",   "ETN",   "PH",
    "6301.T",                                         # Komatsu
    "SIE.DE",                                         # Siemens
    "ABB",                                            # ABB
    # ── Transportation / Shipping ────────────────────────────────────────
    "UPS",   "FDX",   "DAL",   "UAL",   "AAL",   "LUV",
    "ODFL",  "JBHT",  "ZIM",   "MATX",
    "MAERSK-B.CO",                                    # Maersk (Denmark)
    "9101.T",                                         # NYK Line (Japan)
    "2866.HK",                                        # COSCO Shipping
    # ── Construction ─────────────────────────────────────────────────────
    "DHI",   "LEN",   "NVR",   "TOL",
    "2914.HK",                                        # CR Construction
    # ── Materials / Mining ───────────────────────────────────────────────
    "BHP",   "RIO",   "FCX",   "NEM",   "VALE",  "AA",
    "LIN",   "APD",   "SHW",   "ECL",
    # ── Chemicals ────────────────────────────────────────────────────────
    "DOW",   "LYB",   "CE",    "EMN",
    "BASFY",                                          # BASF
    # ── Business Services ────────────────────────────────────────────────
    "ACN",   "IBM",   "CTSH",  "WIT",   "INFO",  "VRSK",
    # ── Hotels / Leisure ─────────────────────────────────────────────────
    "MAR",   "HLT",   "IHG",   "WYNN",  "MGM",   "LVS",
    # ── Real Estate (REITs → Diversified Financials) ─────────────────────
    "PLD",   "AMT",   "SPG",   "EQIX",  "O",
]

# ---------------------------------------------------------------------------
# YF industry → Forbes category (same as app.py)
# ---------------------------------------------------------------------------

YF_INDUSTRY_MAP = {
    "Software—Application":                "IT Software & Services",
    "Software—Infrastructure":             "IT Software & Services",
    "Software - Application":              "IT Software & Services",
    "Software - Infrastructure":           "IT Software & Services",
    "Information Technology Services":     "IT Software & Services",
    "Internet Content & Information":      "IT Software & Services",
    "Semiconductors":                      "Semiconductors",
    "Semiconductor Equipment & Materials": "Semiconductors",
    "Computer Hardware":                   "Technology Hardware & Equipment",
    "Consumer Electronics":                "Technology Hardware & Equipment",
    "Electronic Components":               "Technology Hardware & Equipment",
    "Communication Equipment":             "Technology Hardware & Equipment",
    "Drug Manufacturers—General":          "Drugs & Biotechnology",
    "Drug Manufacturers—Specialty & Generic": "Drugs & Biotechnology",
    "Biotechnology":                       "Drugs & Biotechnology",
    "Medical Devices":                     "Health Care Equipment & Services",
    "Healthcare Plans":                    "Health Care Equipment & Services",
    "Medical Care Facilities":             "Health Care Equipment & Services",
    "Diagnostics & Research":              "Health Care Equipment & Services",
    "Banks—Diversified":                   "Banking",
    "Banks—Regional":                      "Banking",
    "Insurance—Diversified":               "Insurance",
    "Insurance—Life":                      "Insurance",
    "Insurance—Property & Casualty":       "Insurance",
    "Asset Management":                    "Diversified Financials",
    "Capital Markets":                     "Diversified Financials",
    "Financial Data & Stock Exchanges":    "Diversified Financials",
    "Credit Services":                     "Diversified Financials",
    "Oil & Gas Integrated":                "Oil & Gas Operations",
    "Oil & Gas E&P":                       "Oil & Gas Operations",
    "Oil & Gas Midstream":                 "Oil & Gas Operations",
    "Oil & Gas Refining & Marketing":      "Oil & Gas Operations",
    "Oil & Gas Equipment & Services":      "Oil & Gas Operations",
    "Aerospace & Defense":                 "Aerospace & Defense",
    "Farm & Heavy Construction Machinery": "Capital Goods",
    "Industrial Machinery & Supplies & Components": "Capital Goods",
    "Electrical Equipment & Parts":        "Capital Goods",
    "Building Products & Equipment":       "Capital Goods",
    "Marine Shipping":                     "Transportation",
    "Trucking":                            "Transportation",
    "Airlines":                            "Transportation",
    "Railroads":                           "Transportation",
    "Airport Services":                    "Transportation",
    "Integrated Shipping & Logistics":     "Transportation",
    "Courier & Delivery Services":         "Transportation",
    "Internet Retail":                     "Retailing",
    "Specialty Retail":                    "Retailing",
    "Department Stores":                   "Retailing",
    "Discount Stores":                     "Retailing",
    "Home Improvement Retail":             "Retailing",
    "Grocery Stores":                      "Food Markets",
    "Food Distribution":                   "Food, Drink & Tobacco",
    "Packaged Foods":                      "Food, Drink & Tobacco",
    "Beverages—Non-Alcoholic":             "Food, Drink & Tobacco",
    "Beverages—Alcoholic":                 "Food, Drink & Tobacco",
    "Tobacco":                             "Food, Drink & Tobacco",
    "Telecom Services":                    "Telecommunications Services",
    "Entertainment":                       "Media",
    "Broadcasting":                        "Media",
    "Publishing":                          "Media",
    "Steel":                               "Materials",
    "Aluminum":                            "Materials",
    "Copper":                              "Materials",
    "Gold":                                "Materials",
    "Specialty Chemicals":                 "Chemicals",
    "Agricultural Inputs":                 "Chemicals",
    "Diversified Chemicals":               "Chemicals",
    "Engineering & Construction":          "Construction",
    "Utilities—Regulated Electric":        "Utilities",
    "Utilities—Regulated Gas":             "Utilities",
    "Utilities—Diversified":               "Utilities",
    "Utilities—Renewable":                 "Utilities",
    "Hotels & Motels":                     "Hotels, Restaurants & Leisure",
    "Restaurants":                         "Hotels, Restaurants & Leisure",
    "Leisure":                             "Hotels, Restaurants & Leisure",
    "Gambling":                            "Hotels, Restaurants & Leisure",
    "Household & Personal Products":       "Household & Personal Products",
    "Personal Services":                   "Business Services & Supplies",
    "Staffing & Employment Services":      "Business Services & Supplies",
    "Rental & Leasing Services":           "Business Services & Supplies",
    "Conglomerates":                       "Conglomerates",
    "REIT—Retail":                         "Diversified Financials",
    "REIT—Office":                         "Diversified Financials",
    "REIT—Industrial":                     "Diversified Financials",
    "REIT—Diversified":                    "Diversified Financials",
    "Real Estate Services":                "Business Services & Supplies",
}

SECTOR_MAP = {
    "Technology":             "IT Software & Services",
    "Healthcare":             "Drugs & Biotechnology",
    "Financial Services":     "Diversified Financials",
    "Consumer Cyclical":      "Retailing",
    "Consumer Defensive":     "Food, Drink & Tobacco",
    "Energy":                 "Oil & Gas Operations",
    "Industrials":            "Capital Goods",
    "Basic Materials":        "Materials",
    "Communication Services": "Telecommunications Services",
    "Real Estate":            "Diversified Financials",
    "Utilities":              "Utilities",
}

COUNTRY_REGION = {
    "United States": "North America", "Canada": "North America",
    "Mexico": "Latin America", "Brazil": "Latin America",
    "Argentina": "Latin America", "Chile": "Latin America",
    "United Kingdom": "Europe", "Germany": "Europe", "France": "Europe",
    "Italy": "Europe", "Spain": "Europe", "Netherlands": "Europe",
    "Switzerland": "Europe", "Sweden": "Europe", "Norway": "Europe",
    "Denmark": "Europe", "Finland": "Europe", "Belgium": "Europe",
    "Austria": "Europe", "Poland": "Europe", "Russia": "Europe",
    "Portugal": "Europe", "Ireland": "Europe", "Luxembourg": "Europe",
    "Turkey": "Europe",
    "China": "Asia-Pacific", "Japan": "Asia-Pacific",
    "South Korea": "Asia-Pacific", "Taiwan": "Asia-Pacific",
    "Hong Kong": "Asia-Pacific", "India": "Asia-Pacific",
    "Singapore": "Asia-Pacific", "Thailand": "Asia-Pacific",
    "Malaysia": "Asia-Pacific", "Indonesia": "Asia-Pacific",
    "Australia": "Asia-Pacific", "New Zealand": "Asia-Pacific",
    "Saudi Arabia": "Middle East & Africa",
    "United Arab Emirates": "Middle East & Africa",
    "Qatar": "Middle East & Africa", "Israel": "Middle East & Africa",
    "South Africa": "Middle East & Africa", "Nigeria": "Middle East & Africa",
    "Egypt": "Middle East & Africa",
}


def fetch_one(ticker: str) -> dict | None:
    try:
        t    = yf.Ticker(ticker)
        info = t.info
        name = info.get("longName") or info.get("shortName")
        if not name:
            return None

        revenue    = info.get("totalRevenue")
        net_income = info.get("netIncomeToCommon")
        market_cap = info.get("marketCap")
        if not revenue or not market_cap or revenue <= 0 or market_cap <= 0:
            return None

        try:
            bs = t.balance_sheet
            total_assets = float(bs.loc["Total Assets"].iloc[0]) \
                           if "Total Assets" in bs.index else None
        except Exception:
            total_assets = None

        # Convert to USD — financials use financialCurrency, market cap uses currency
        trading_currency = info.get("currency", "USD") or "USD"
        fin_currency     = info.get("financialCurrency", "USD") or "USD"
        fx_trading       = get_usd_rate(trading_currency)
        fx_fin           = get_usd_rate(fin_currency)

        sales_B    = (revenue    * fx_fin)     / 1e9
        profit_B   = (net_income * fx_fin)     / 1e9 if net_income else 0.0
        assets_B   = (total_assets * fx_fin)   / 1e9 if total_assets else sales_B * 1.5
        mktcap_B   = (market_cap * fx_trading) / 1e9
        yf_ind     = info.get("industry", "")
        yf_sec     = info.get("sector", "")
        country    = info.get("country", "")
        industry   = (YF_INDUSTRY_MAP.get(yf_ind)
                      or SECTOR_MAP.get(yf_sec)
                      or "IT Software & Services")

        return {
            "ticker":       ticker,
            "Company":      name,
            "Country":      country,
            "Location":     f"{info.get('city', '')}, {country}".strip(", "),
            "region":       COUNTRY_REGION.get(country, "Other"),
            "Industry":     industry,
            "yf_industry":  yf_ind,
            "yf_sector":    yf_sec,
            "Sales_B":      round(sales_B,  3),
            "Profits_B":    round(profit_B, 3),
            "Assets_B":     round(assets_B, 3),
            "MarketValue_B": round(mktcap_B, 3),
            "employees":          info.get("fullTimeEmployees") or 0,
            "description":        info.get("longBusinessSummary", ""),
            "trading_currency":   trading_currency,
            "fin_currency":       fin_currency,
            "fetched_at":         datetime.now().isoformat(),
        }
    except Exception:
        return None


def fetch_all(tickers: list[str], max_workers: int = 10) -> pd.DataFrame:
    rows   = []
    failed = []
    done   = 0
    total  = len(tickers)

    print(f"Fetching {total} tickers with {max_workers} concurrent workers…\n")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_one, t): t for t in tickers}
        for fut in as_completed(futs):
            ticker = futs[fut]
            done  += 1
            try:
                row = fut.result()
                if row:
                    rows.append(row)
                    print(f"  [{done:3d}/{total}] ✓  {ticker:<18} {row['Company'][:30]}")
                else:
                    failed.append(ticker)
                    print(f"  [{done:3d}/{total}] ✗  {ticker} — no data")
            except Exception as e:
                failed.append(ticker)
                print(f"  [{done:3d}/{total}] ✗  {ticker} — {e}")

    df = pd.DataFrame(rows)
    print(f"\n{'='*55}")
    print(f"Fetched:  {len(df)} companies")
    print(f"Failed:   {len(failed)} ({', '.join(failed[:10])}{'…' if len(failed)>10 else ''})")
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["Sales_B"] > 0].copy()
    df["ps_ratio"]      = df["MarketValue_B"] / df["Sales_B"]
    df["profit_margin"] = df["Profits_B"] / df["Sales_B"]
    df["log_sales"]     = np.log1p(df["Sales_B"])
    df["log_assets"]    = np.log1p(df["Assets_B"])
    # Sector medians
    sector_medians          = df.groupby("Industry")["ps_ratio"].median()
    df["sector_median_ps"]  = df["Industry"].map(sector_medians)
    df["value_target"]      = (df["ps_ratio"] < df["sector_median_ps"]).astype(int)
    return df


def main():
    t0  = time.time()
    df  = fetch_all(TICKER_UNIVERSE)
    df  = enrich(df)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "df":           df,
        "fetched_at":   datetime.now().isoformat(),
        "n_companies":  len(df),
        "n_industries": df["Industry"].nunique(),
    }
    with open(OUTPUT, "wb") as f:
        pickle.dump(meta, f)

    elapsed = time.time() - t0
    print(f"\nSaved {len(df)} companies to {OUTPUT}")
    print(f"Industries: {df['Industry'].nunique()}")
    print(f"Regions:    {df['region'].nunique()}")
    print(f"Elapsed:    {elapsed/60:.1f} min")
    print(f"\nTop 5 by market cap:")
    print(df.nlargest(5, "MarketValue_B")[
        ["Company", "Country", "Industry", "Sales_B", "MarketValue_B", "ps_ratio"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
