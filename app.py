import re
import io
import time
import ipaddress
import socket
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pathlib
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Constants & mappings
# ---------------------------------------------------------------------------

NUMERIC_FEATURES     = ["Sales_B", "Assets_B", "Profits_B", "profit_margin",
                         "log_sales", "log_assets",
                         "rev_growth", "ebitda_margin", "gross_margin"]
CATEGORICAL_FEATURES = ["Industry"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES

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
    "Tools & Accessories":                 "Capital Goods",
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
    "Diversified Chemicals":               "Chemicals",
    "Agricultural Inputs":                 "Chemicals",
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

INDUSTRY_LABELS = {
    "IT Software & Services":           "Tech — Software",
    "Semiconductors":                   "Tech — Semiconductors",
    "Technology Hardware & Equipment":  "Tech — Hardware",
    "Drugs & Biotechnology":            "Healthcare / Pharma",
    "Health Care Equipment & Services": "Healthcare — Equipment",
    "Banking":                          "Banking",
    "Diversified Financials":           "Finance — Diversified",
    "Insurance":                        "Insurance",
    "Oil & Gas Operations":             "Energy — Oil & Gas",
    "Utilities":                        "Energy — Utilities",
    "Retailing":                        "Retail / E-commerce",
    "Consumer Durables":                "Consumer Goods",
    "Food, Drink & Tobacco":            "Food & Beverage",
    "Food Markets":                     "Grocery / Food Stores",
    "Telecommunications Services":      "Telecom",
    "Media":                            "Media",
    "Transportation":                   "Transportation / Shipping",
    "Aerospace & Defense":              "Aerospace & Defense",
    "Capital Goods":                    "Manufacturing / Machinery",
    "Business Services & Supplies":     "Business Services",
    "Household & Personal Products":    "Personal Products",
    "Materials":                        "Materials / Mining",
    "Chemicals":                        "Chemicals",
    "Construction":                     "Construction",
    "Hotels, Restaurants & Leisure":    "Hotels / Restaurants",
    "Trading Companies":                "Trading",
    "Conglomerates":                    "Conglomerates",
}

# Groups the 27 Forbes industries above into 7 broad clusters, used to widen
# the comparable-companies peer pool beyond exact-industry matches (see
# score_comparables()). Keys must exactly match the "Industry" column values
# — including punctuation (commas) — or the cluster lookup silently misses.
INDUSTRY_CLUSTERS = {
    "IT Software & Services":           "Technology",
    "Semiconductors":                   "Technology",
    "Technology Hardware & Equipment":  "Technology",
    "Drugs & Biotechnology":            "Healthcare",
    "Health Care Equipment & Services": "Healthcare",
    "Banking":                          "Financials",
    "Diversified Financials":           "Financials",
    "Insurance":                        "Financials",
    "Oil & Gas Operations":             "Energy & Utilities",
    "Utilities":                        "Energy & Utilities",
    "Retailing":                        "Consumer",
    "Consumer Durables":                "Consumer",
    "Food, Drink & Tobacco":            "Consumer",
    "Food Markets":                     "Consumer",
    "Household & Personal Products":    "Consumer",
    "Hotels, Restaurants & Leisure":    "Consumer",
    "Telecommunications Services":      "Telecom & Media",
    "Media":                            "Telecom & Media",
    "Transportation":                   "Industrials & Materials",
    "Aerospace & Defense":              "Industrials & Materials",
    "Capital Goods":                    "Industrials & Materials",
    "Business Services & Supplies":     "Industrials & Materials",
    "Materials":                        "Industrials & Materials",
    "Chemicals":                        "Industrials & Materials",
    "Construction":                     "Industrials & Materials",
    "Trading Companies":                "Industrials & Materials",
    "Conglomerates":                    "Industrials & Materials",
}

COUNTRY_REGION = {
    "United States": "North America", "Canada": "North America",
    "Mexico": "Latin America", "Brazil": "Latin America",
    "Argentina": "Latin America", "Chile": "Latin America",
    "Colombia": "Latin America", "Peru": "Latin America",
    "United Kingdom": "Europe", "Germany": "Europe", "France": "Europe",
    "Italy": "Europe", "Spain": "Europe", "Netherlands": "Europe",
    "Switzerland": "Europe", "Sweden": "Europe", "Norway": "Europe",
    "Denmark": "Europe", "Finland": "Europe", "Belgium": "Europe",
    "Austria": "Europe", "Poland": "Europe", "Russia": "Europe",
    "Portugal": "Europe", "Ireland": "Europe", "Luxembourg": "Europe",
    "Czech Republic": "Europe", "Greece": "Europe", "Turkey": "Europe",
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

VERDICT_STYLE = {
    "UNDERVALUED": {"bg": "#d4edda", "text": "#155724", "emoji": "🟢"},
    "OVERVALUED":  {"bg": "#f8d7da", "text": "#721c24", "emoji": "🔴"},
    "FAIR VALUE":  {"bg": "#fff3cd", "text": "#856404", "emoji": "🟡"},
    # No market price to compare against (private/manual company) — an
    # estimate, not an undervalued/overvalued/fair-value judgment.
    "ESTIMATE":    {"bg": "#e2e3e5", "text": "#383d41", "emoji": "📊"},
}
VERDICT_DISPLAY_LABEL = {"ESTIMATE": "ESTIMATED VALUATION"}
# Weight of the trained model's own regression estimate vs. the peer median
# in the blended fair-value/estimate benchmark (see blended_benchmark below).
MODEL_WEIGHT = 0.30
PEER_WEIGHT  = 0.70

# ---------------------------------------------------------------------------
# Data loading  ── Forbes = training + comps pool (always)
#               ── YF cache = live lookups + metadata only
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading Forbes Global 2000…")
def load_training_data(enriched_mtime: float = 0.0, xlsx_mtime: float = 0.0):
    """
    Load Forbes Global 2000 as ML training + comps pool.
    Uses the description-enriched version (forbes_enriched.pkl) when available;
    falls back to plain forbes_global2000.xlsx otherwise.
    Returns (df, sector_medians, has_descriptions, source_label).
    """
    enriched = pathlib.Path("data/forbes_enriched.pkl")
    if enriched.exists():
        with open(enriched, "rb") as f:
            meta = pickle.load(f)
        df        = meta["df"]
        n_desc    = meta.get("n_with_desc", (df["description"].str.len() > 50).sum())
        has_descs = True
        source    = (f"Forbes G2000 + descriptions "
                     f"({n_desc:,}/{len(df):,} enriched, "
                     f"{meta.get('coverage_pct','?')}% coverage)")
    else:
        df        = pd.read_excel("data/forbes_global2000.xlsx")
        has_descs = False
        source    = "Forbes Global 2000 (plain — run enrich_forbes_descriptions.py for descriptions)"

    df = df.dropna(subset=["Sales_B"]).copy()
    df = df[df["Sales_B"] > 0].copy()

    # EV/Revenue is the primary multiple.
    # For non-financial companies: EV = MarketCap + NetDebt (when net_debt_B available).
    # For banks, insurers, and diversified financials: yfinance counts customer deposits
    # as "total debt", making EV negative and meaningless. Use MarketCap as EV proxy
    # (equivalent to P/S) for these sectors — P/B is the proper metric there anyway.
    FINANCIAL_SECTORS = {"Banking", "Insurance", "Diversified Financials"}
    if "net_debt_B" in df.columns:
        net_debt = df["net_debt_B"].fillna(0).copy()
        net_debt[df["Industry"].isin(FINANCIAL_SECTORS)] = 0
        df["ev_B"] = df["MarketValue_B"] + net_debt
    else:
        df["ev_B"] = df["MarketValue_B"]
    df["ev_revenue"]     = df["ev_B"] / df["Sales_B"]
    df["profit_margin"]  = df["Profits_B"] / df["Sales_B"]
    df["log_sales"]      = np.log1p(df["Sales_B"])
    df["log_assets"]     = np.log1p(df["Assets_B"])
    if "region" not in df.columns:
        df["region"] = df["Country"].map(COUNTRY_REGION).fillna("Other")
    # Ensure new financial feature columns exist (NaN when not enriched;
    # SimpleImputer(strategy="median") handles missing values in the pipeline).
    for col in ["rev_growth", "ebitda_margin", "gross_margin"]:
        if col not in df.columns:
            df[col] = np.nan

    # Derived EBITDA absolute $ and EV/EBITDA multiple — only defined where
    # ebitda_margin is available (~48% coverage); NaN elsewhere.
    df["ebitda_B"]   = df["ebitda_margin"] * df["Sales_B"]
    df["ev_ebitda"]  = np.where(df["ebitda_B"] > 0, df["ev_B"] / df["ebitda_B"], np.nan)

    # Remove any remaining outliers (near-zero revenue, bad YF data).
    n_before = len(df)
    df = df[df["ev_revenue"].between(0.01, 200)].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        source += f" · {n_dropped} EV/Rev outliers removed"

    sector_medians          = df.groupby("Industry")["ev_revenue"].median()
    df["sector_median_ev"]  = df["Industry"].map(sector_medians)
    df["value_target"]      = (df["ev_revenue"] < df["sector_median_ev"]).astype(int)
    return df, sector_medians, has_descs, source


@st.cache_resource(show_spinner=False)
def build_tfidf_index(enriched_mtime: float = 0.0):
    """
    Build TF-IDF index from Forbes company descriptions.
    Returns (vectorizer, matrix, df_index_list) or (None, None, None)
    if descriptions are not yet available.
    """
    enriched = pathlib.Path("data/forbes_enriched.pkl")
    if not enriched.exists():
        return None, None, None
    with open(enriched, "rb") as f:
        meta = pickle.load(f)
    df    = meta["df"]
    texts = df["description"].fillna("").tolist()
    if sum(len(t) for t in texts) < 10_000:
        return None, None, None
    vec    = TfidfVectorizer(max_features=1500, stop_words="english",
                             min_df=2, ngram_range=(1, 2))
    matrix = vec.fit_transform(texts)
    return vec, matrix, df.index.tolist()


@st.cache_data(show_spinner=False)
def load_yf_cache_info() -> dict | None:
    """Metadata about the local Yahoo Finance cache (if present)."""
    cache = pathlib.Path("data/yf_training_data.pkl")
    if not cache.exists():
        return None
    try:
        with open(cache, "rb") as f:
            meta = pickle.load(f)
        return {
            "n":    meta.get("n_companies", len(meta["df"])),
            "date": meta.get("fetched_at", "")[:10],
        }
    except Exception:
        return None

# ---------------------------------------------------------------------------
# ML pipeline
# ---------------------------------------------------------------------------

def build_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("cat", Pipeline([
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_FEATURES),
    ])


@st.cache_resource(show_spinner="Training valuation model…")
def train_models(enriched_mtime: float = 0.0, xlsx_mtime: float = 0.0):
    df, sector_medians, _, _ = load_training_data(enriched_mtime, xlsx_mtime)
    X = df[ALL_FEATURES]

    # Hyperparameters chosen to minimise overfitting (CV gap 0.11 vs 0.70 before):
    #   subsample=0.8  → stochastic boosting, reduces variance
    #   min_samples_leaf=10 → prevents deep splits on small groups
    #   max_depth=3 (was 4) → less complex trees
    GB_PARAMS = dict(n_estimators=150, max_depth=3, learning_rate=0.05,
                     subsample=0.8, min_samples_leaf=10, random_state=42)

    clf = Pipeline([("pre", build_preprocessor()),
                    ("m", GradientBoostingClassifier(**GB_PARAMS))])
    reg = Pipeline([("pre", build_preprocessor()),
                    ("m", GradientBoostingRegressor(**GB_PARAMS))])

    clf.fit(X, df["value_target"])
    # Train on log1p(ev_revenue): EV/Revenue is right-skewed; log-space
    # raises CV R² from 0.10 to 0.45 and cuts the overfitting gap from 0.70 to 0.11.
    reg.fit(X, np.log1p(df["ev_revenue"]))

    # Second regressor: EV/EBITDA, trained only on the subset with EBITDA
    # coverage (~48% of the pool) — a smaller, noisier training set than the
    # primary EV/Revenue model, so its predictions are lower-confidence.
    df_ebitda = df[df["ev_ebitda"].notna()]
    reg_ebitda = Pipeline([("pre", build_preprocessor()),
                           ("m", GradientBoostingRegressor(**GB_PARAMS))])
    reg_ebitda.fit(df_ebitda[ALL_FEATURES], np.log1p(df_ebitda["ev_ebitda"]))

    return clf, reg, reg_ebitda, sector_medians


@st.cache_data(show_spinner=False)
def industry_feature_medians(enriched_mtime: float = 0.0, xlsx_mtime: float = 0.0):
    """Per-industry median for every NUMERIC_FEATURES column — used by explain_verdict()."""
    df, _, _, _ = load_training_data(enriched_mtime, xlsx_mtime)
    return df.groupby("Industry")[NUMERIC_FEATURES].median()


# Numeric inputs that are true independent degrees of freedom. log_sales, log_assets,
# and profit_margin are deterministic transforms of Sales_B/Assets_B/Profits_B — they
# are never independently perturbed or ranked, only recomputed as a side effect below.
INDEPENDENT_FEATURES = ["Sales_B", "Assets_B", "Profits_B",
                        "rev_growth", "ebitda_margin", "gross_margin"]
FEATURE_DISPLAY = {
    "Sales_B":       "Revenue",
    "Assets_B":      "Total assets",
    "Profits_B":     "Net profit",
    "rev_growth":    "Revenue growth",
    "ebitda_margin": "EBITDA margin",
    "gross_margin":  "Gross margin",
}


def explain_verdict(reg, input_df: pd.DataFrame, industry: str,
                    medians: pd.DataFrame) -> list[dict]:
    """
    Rank the independent inputs by how much resetting each one (alone) to its
    industry median shifts the model's predicted EV/Revenue for this company.
    Positive shift = this company's actual value pulls the prediction above what
    an industry-median peer would get on that feature alone.
    Skips features the input row has no value for (imputed inputs carry no
    company-specific story to tell) and industries with no median on record.
    """
    if industry not in medians.index:
        return []
    med_row  = medians.loc[industry]
    base_pred = float(np.expm1(reg.predict(input_df)[0]))

    results = []
    for feat in INDEPENDENT_FEATURES:
        actual = input_df.iloc[0][feat]
        if pd.isna(actual):
            continue
        med_val = med_row[feat]
        if pd.isna(med_val):
            continue

        perturbed = input_df.copy()
        idx = perturbed.index[0]
        perturbed.loc[idx, feat] = med_val
        # Recompute whichever derived columns depend on the perturbed raw feature,
        # so the perturbed row stays internally consistent (see plan KTD).
        if feat == "Sales_B":
            s = max(float(med_val), 0.001)
            perturbed.loc[idx, "log_sales"]     = np.log1p(s)
            perturbed.loc[idx, "profit_margin"] = float(perturbed.loc[idx, "Profits_B"]) / s
        elif feat == "Assets_B":
            perturbed.loc[idx, "log_assets"] = np.log1p(float(med_val))
        elif feat == "Profits_B":
            s = max(float(perturbed.loc[idx, "Sales_B"]), 0.001)
            perturbed.loc[idx, "profit_margin"] = float(med_val) / s

        perturbed_pred = float(np.expm1(reg.predict(perturbed)[0]))
        results.append({
            "feature": feat,
            "label":   FEATURE_DISPLAY[feat],
            "actual":  float(actual),
            "median":  float(med_val),
            "shift":   base_pred - perturbed_pred,
        })

    results.sort(key=lambda r: abs(r["shift"]), reverse=True)
    return results

# ---------------------------------------------------------------------------
# Yahoo Finance live data
# ---------------------------------------------------------------------------

class YFTemporaryError(Exception):
    """Transient Yahoo Finance failure (rate limit / crumb expiry / network).

    Raised (never returned) so @st.cache_data does not cache the failure —
    a rate-limited query can be retried immediately once Yahoo recovers,
    instead of being stuck returning empty/None for the full cache ttl.
    """


def _yf_call_with_retry(fn, retries: int = 3, backoff: float = 0.6):
    """Run fn() with retries on transient errors; raise YFTemporaryError if
    all attempts fail. Real "no data" responses (empty list, missing keys)
    are returned as-is and are NOT retried."""
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise YFTemporaryError(str(last_exc)) from last_exc


@st.cache_data(ttl=3600, show_spinner=False)
def search_companies(query: str) -> list[dict]:
    if not query or len(query) < 2:
        return []
    quotes = _yf_call_with_retry(lambda: yf.Search(query, max_results=8).quotes)
    return [
        {"label": f"{r.get('longname') or r.get('shortname', r['symbol'])}  "
                  f"({r['symbol']}) · {r.get('exchange', '')}",
         "symbol": r["symbol"]}
        for r in quotes if r.get("symbol")
    ]


@st.cache_data(ttl=3600, show_spinner=False)
def get_usd_rate(currency: str) -> float:
    """Return the USD value of 1 unit of `currency` (e.g. 1 JPY = 0.0067 USD)."""
    if not currency or currency == "USD":
        return 1.0
    try:
        fx   = yf.Ticker(f"{currency}USD=X")
        rate = fx.fast_info.last_price
        if not rate or rate <= 0:
            hist = fx.history(period="1d")
            rate = float(hist["Close"].iloc[-1]) if not hist.empty else 1.0
        return float(rate)
    except Exception:
        return 1.0


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_usd_rate(currency: str) -> float:
    return get_usd_rate(currency)


def fetch_ticker(ticker: str) -> dict | None:
    t = yf.Ticker(ticker.upper().strip())
    info = _yf_call_with_retry(lambda: t.info)

    if not info.get("shortName"):
        return None  # ticker genuinely doesn't exist / no coverage

    revenue    = info.get("totalRevenue")
    net_income = info.get("netIncomeToCommon")
    market_cap = info.get("marketCap")
    if not revenue or not market_cap or revenue <= 0:
        return None  # genuinely missing fundamentals (not a fetch failure)

    # Convert to USD — market_cap is denominated in the trading currency,
    # financial-statement fields in the (possibly different) reporting
    # currency. Collapsing these into a single rate would misconvert
    # market_cap whenever the two diverge (ADRs, foreign private issuers).
    trading_currency = info.get("currency", "USD") or "USD"
    fin_currency      = info.get("financialCurrency", "USD") or "USD"
    fx_trading         = _cached_usd_rate(trading_currency)
    fx_fin              = _cached_usd_rate(fin_currency)

    revenue      = revenue * fx_fin
    net_income   = net_income * fx_fin if net_income is not None else None
    market_cap   = market_cap * fx_trading

    try:
        bs = t.balance_sheet
        total_assets = float(bs.loc["Total Assets"].iloc[0]) \
                       if "Total Assets" in bs.index else None
        if total_assets is not None:
            total_assets *= fx_fin
    except Exception:
        total_assets = None

    sales_B  = revenue / 1e9
    profit_B = net_income / 1e9 if net_income is not None else 0.0
    assets_B = total_assets / 1e9 if total_assets else sales_B * 1.5
    mktcap_B = market_cap / 1e9

    yf_ind   = info.get("industry", "")
    yf_sec   = info.get("sector", "")
    country  = info.get("country", "")
    industry = (YF_INDUSTRY_MAP.get(yf_ind)
                or SECTOR_MAP.get(yf_sec)
                or "IT Software & Services")

    # For financial companies yfinance includes customer deposits in totalDebt,
    # producing nonsensical (often negative) EV. Zero out net debt for these
    # sectors — same convention used in the Forbes training pool.
    FINANCIAL_SECTORS = {"Banking", "Insurance", "Diversified Financials"}
    is_financial = industry in FINANCIAL_SECTORS
    if is_financial:
        net_debt_B = 0.0
        ev_B       = mktcap_B
    else:
        total_debt = ((info.get("totalDebt") or 0) * fx_fin) / 1e9
        total_cash = ((info.get("totalCash") or 0) * fx_fin) / 1e9
        net_debt_B = total_debt - total_cash
        ev_B       = mktcap_B + net_debt_B

    ebitda_raw  = info.get("ebitda")
    ebitda_raw  = ebitda_raw * fx_fin if ebitda_raw else ebitda_raw
    ebitda_B    = ebitda_raw / 1e9 if ebitda_raw else None
    ebitda_marg = ebitda_B / sales_B if ebitda_B and sales_B > 0 else None
    net_margin  = info.get("profitMargins")
    rev_growth  = info.get("revenueGrowth")
    gross_marg  = info.get("grossMargins")
    fcf_raw     = info.get("freeCashflow")
    fcf_raw     = fcf_raw * fx_fin if fcf_raw else fcf_raw
    fcf_margin  = (fcf_raw / revenue) if fcf_raw and revenue else None
    ev_revenue  = ev_B / sales_B if sales_B > 0 else None
    ev_ebitda   = ev_B / ebitda_B if ebitda_B and ebitda_B > 0 else None
    ps_actual   = mktcap_B / sales_B if sales_B > 0 else None
    desc = info.get("longBusinessSummary", "")

    return {
        "name":          info.get("shortName", ticker.upper()),
        "ticker":        ticker.upper().strip(),
        "country":       country,
        "region":        COUNTRY_REGION.get(country, "Other"),
        "industry":      industry,
        "yf_industry":   yf_ind,
        "yf_sector":     yf_sec,
        "sales_B":       sales_B,
        "profit_B":      profit_B,
        "assets_B":      assets_B,
        "ebitda_B":      ebitda_B,
        "net_debt_B":    net_debt_B,
        "mktcap_B":      mktcap_B,
        "ev_B":          ev_B,
        "ps_actual":     ps_actual,
        "ev_revenue":    ev_revenue,
        "ev_ebitda":     ev_ebitda,
        "pe_ratio":      info.get("trailingPE"),
        "pb_ratio":      info.get("priceToBook"),
        "eps":           info.get("trailingEps"),
        "ebitda_margin": ebitda_marg,
        "gross_margin":  info.get("grossMargins"),
        "net_margin":    net_margin,
        "rev_growth":    rev_growth,
        "description":   (desc[:500] + "…") if len(desc) > 500 else desc,
        "fetched_at":    datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_price_history(ticker: str) -> pd.DataFrame | None:
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        return hist.reset_index() if not hist.empty else None
    except Exception:
        return None


_SCRAPE_MAX_BYTES = 2_000_000  # response-size cap before parsing


def _is_safe_url(url: str) -> tuple[bool, str]:
    """
    Reject non-http(s) schemes and private/loopback/link-local/multicast
    targets (including cloud metadata addresses) before any request is
    issued. This is the app's first outbound fetch driven by arbitrary
    pasted user input, unlike the ticker-only yfinance calls elsewhere.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Could not parse that as a URL."
    if parsed.scheme not in ("http", "https"):
        return False, "Only http:// and https:// URLs are supported."
    if not parsed.hostname:
        return False, "URL is missing a host."
    try:
        addrs = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        return False, "Could not resolve that host."
    for _family, _type, _proto, _canon, sockaddr in addrs:
        ip = ipaddress.ip_address(sockaddr[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
            return False, "That URL points to a private or internal address, which isn't allowed."
    return True, ""


def scrape_description_from_url(url: str) -> tuple[str | None, str | None]:
    """
    Fetch a company website and extract visible text for the description
    field. Returns (text, None) on success, or (None, reason) on any failure
    — the caller shows the reason as a warning and leaves the existing
    description untouched, never crashing the page.
    """
    safe, reason = _is_safe_url(url)
    if not safe:
        return None, reason

    try:
        resp = requests.get(
            url, timeout=8,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ValuatorixBot/1.0)"},
            allow_redirects=False, stream=True,
        )
    except requests.exceptions.Timeout:
        return None, "The request timed out."
    except requests.exceptions.RequestException:
        return None, "Could not reach that URL."

    if resp.status_code != 200:
        return None, f"That page returned HTTP {resp.status_code}."

    content = b""
    for chunk in resp.iter_content(chunk_size=65536):
        content += chunk
        if len(content) > _SCRAPE_MAX_BYTES:
            return None, "That page's response was too large to process."

    try:
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
    except Exception:
        return None, "Could not parse that page's content."

    if not text:
        return None, "No readable text was found on that page."

    return text[:2000], None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_company_news(ticker: str) -> list[dict]:
    """
    Recent headlines for a ticker, via yfinance's news feed. Unlike
    search_companies()/fetch_ticker(), failures here are swallowed rather than
    raised — this is a secondary enrichment on an already-loaded company, not
    the user's primary action, so a Yahoo Finance hiccup shouldn't interrupt or
    clutter the "Why this verdict?" section. Returns [] on any failure or when
    there's simply no news.
    """
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []

    headlines = []
    for item in items[:5]:
        content = item.get("content", {})
        title   = content.get("title")
        if not title:
            continue
        headlines.append({
            "title":     title,
            "pubDate":   content.get("pubDate", ""),
            "publisher": (content.get("provider") or {}).get("displayName", ""),
            "url":       (content.get("canonicalUrl") or {}).get("url", ""),
        })
    return headlines

# ---------------------------------------------------------------------------
# Comparable company scoring  (Forbes pool, 3-dimensional)
# ---------------------------------------------------------------------------

def _clean_name(name: str) -> str:
    """Strip common legal suffixes for fuzzy company-name matching."""
    n = name.lower().strip()
    for suffix in [
        " incorporated", " corporation", " holdings", " holding",
        " limited", " group", " company", " plc", " inc.", " inc",
        " corp.", " corp", " ltd.", " ltd", " llc", " ag", " nv",
        " sa", " s.a.", " co.", " co", " a/s", " asa", " ab",
    ]:
        n = n.replace(suffix, "")
    return n.strip(" .,")


def _is_same_company(forbes_name: str, target_name: str) -> bool:
    """Return True if the two names most likely refer to the same company."""
    a = _clean_name(forbes_name)
    b = _clean_name(target_name)
    if not a or not b:
        return False
    # Exact match after cleaning
    if a == b:
        return True
    # Shorter name must appear at a word boundary in the longer name
    # ("Apple" in "Apple Inc." ✓  but "Apple" in "Snapple" ✗)
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return len(shorter) >= 4 and bool(re.search(r'\b' + re.escape(shorter), longer))


def score_comparables(
    df: pd.DataFrame,
    industry: str,
    sales_B: float,
    profit_margin: float,
    target_region: str,
    exclude_name: str = "",
    target_description: str = "",
    tfidf_data: tuple = (None, None, None),
    ebitda_margin: float = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Score Forbes companies on up to 5 dimensions:
      With description:    Revenue 10% · Margin 5% · Geography 15% · Description 50% · Industry 20%
      Without description: Revenue 20% · Margin 10% · Geography 30% · Industry 40%

    Description (TF-IDF cosine similarity) carries the most weight when active —
    it captures sub-industry nuance that broad Forbes labels miss
    (e.g. 'SaaS / recurring revenue' vs 'IT distribution' within IT Software).
    The peer pool includes the target's exact industry plus other industries in
    the same INDUSTRY_CLUSTERS group (e.g. Semiconductors for an IT Software
    target), with same-cluster-different-industry peers scored down via the
    industry dimension rather than excluded outright.
    The target company itself is always excluded from the pool.
    """
    target_cluster = INDUSTRY_CLUSTERS.get(industry)
    pool = df[df["Industry"].map(INDUSTRY_CLUSTERS) == target_cluster].copy()
    if len(pool) < 3:
        pool = df.copy()

    # Exclude the target company
    if exclude_name:
        pool = pool[~pool["Company"].apply(lambda n: _is_same_company(n, exclude_name))]

    # ── TF-IDF description similarities ──────────────────────────────────
    vec, matrix, idx_list = tfidf_data
    use_desc = (vec is not None
                and target_description
                and len(target_description) > 80
                and "description" in pool.columns)

    desc_sims = np.zeros(len(pool))
    if use_desc:
        try:
            target_vec  = vec.transform([target_description])
            all_sims    = cosine_similarity(target_vec, matrix)[0]
            # Map pool's DataFrame index positions into TF-IDF matrix positions
            for j, df_idx in enumerate(pool.index):
                if df_idx in idx_list:
                    desc_sims[j] = all_sims[idx_list.index(df_idx)]
        except Exception:
            use_desc = False

    # ── Weights ───────────────────────────────────────────────────────────
    # Description carries the most weight: captures sub-industry nuance that
    # broad Forbes categories miss (SaaS vs IT distribution, toll roads vs airlines).
    if use_desc:
        w_rev, w_mrg, w_geo, w_dsc, w_ind = 0.10, 0.05, 0.15, 0.50, 0.20
    else:
        w_rev, w_mrg, w_geo, w_dsc, w_ind = 0.20, 0.10, 0.30, 0.00, 0.40

    # ── Score every company ───────────────────────────────────────────────
    scores = []
    for j, (_, row) in enumerate(pool.iterrows()):
        if row["Sales_B"] > 0 and sales_B > 0:
            rev_sim = min(row["Sales_B"], sales_B) / max(row["Sales_B"], sales_B)
        else:
            rev_sim = 0.0

        profit_margin_diff = abs((row.get("profit_margin") or 0) - (profit_margin or 0))
        profit_margin_sim  = max(0.0, 1.0 - profit_margin_diff * 3.33)
        row_ebitda_margin  = row.get("ebitda_margin")
        if ebitda_margin is not None and pd.notna(row_ebitda_margin):
            ebitda_margin_diff = abs(row_ebitda_margin - ebitda_margin)
            ebitda_margin_sim  = max(0.0, 1.0 - ebitda_margin_diff * 3.33)
            margin_sim = (profit_margin_sim + ebitda_margin_sim) / 2
        else:
            margin_sim = profit_margin_sim

        geo_sim     = 1.0 if (row.get("region") == target_region
                              and target_region not in ("Other", "")) else 0.0
        industry_sim = 1.0 if row["Industry"] == industry else 0.5

        total = (w_rev * rev_sim + w_mrg * margin_sim + w_geo * geo_sim
                 + w_dsc * float(desc_sims[j]) + w_ind * industry_sim)
        scores.append(round(total * 100))

    pool = pool.copy()
    pool["_score"] = scores
    result = pool.sort_values("_score", ascending=False).head(20)

    ind_label      = INDUSTRY_LABELS.get(industry, industry)
    n_industry     = len(df[df["Industry"] == industry])
    n_cluster      = len(pool)
    cluster_peers  = sorted(set(INDUSTRY_LABELS.get(i, i)
                                 for i in pool["Industry"].unique() if i != industry))
    rationale  = [
        f"**Industry filter:** {ind_label} — {n_industry} companies in Forbes Global 2000 "
        f"(plus {n_cluster - n_industry} from related industries in the same cluster).",
        f"**Revenue size ({int(w_rev*100)}% weight):** Ranked by closeness to **{fmt_B(sales_B)}**. "
        f"Identical size = 100%; 10× different ≈ 10%.",
        f"**Margin similarity ({int(w_mrg*100)}% weight):** Profit margin ≈ **{profit_margin*100:.1f}%**"
        + (f", EBITDA margin ≈ **{ebitda_margin*100:.1f}%**" if ebitda_margin is not None else "")
        + f". Similar margins imply similar cost structure. Score → 0 at ±30 pp gap per component.",
        f"**Geography ({int(w_geo*100)}% weight):** **{target_region}** companies get a bonus. "
        f"Different regions carry different cost-of-capital and growth expectations.",
        f"**Industry classification ({int(w_ind*100)}% weight):** Exact match to **{ind_label}** "
        f"scores 1.0; a same-cluster peer (" + (", ".join(cluster_peers) if cluster_peers else "none in this pool")
        + ") scores 0.5 — captures related-industry comparables without excluding them outright.",
    ]
    if use_desc:
        rationale.append(
            f"**Business description similarity ({int(w_dsc*100)}% weight — TF-IDF):** "
            f"The target's business description is compared to all Forbes company "
            f"descriptions using TF-IDF cosine similarity. This captures nuance "
            f"that broad industry labels miss — e.g. 'SaaS / recurring revenue' vs "
            f"'IT distribution / hardware reselling' within the same Forbes category."
        )
    else:
        rationale.append(
            f"**Description similarity: not active.** "
            f"Run `python scripts/enrich_forbes_descriptions.py` once (~15 min) "
            f"to add business descriptions to all Forbes companies and unlock "
            f"50% description-based scoring for much more accurate comparable selection."
        )
    rationale.append(
        f"**Result:** Top {len(result)} comparables shown, sorted by composite score (0–100)."
    )
    return result, rationale

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_input_row(sales_B, assets_B, profits_B, industry,
                    rev_growth=None, ebitda_margin=None, gross_margin=None):
    s = max(float(sales_B), 0.001)
    return pd.DataFrame([{
        "Sales_B":       s,
        "Assets_B":      float(assets_B),
        "Profits_B":     float(profits_B),
        "profit_margin": float(profits_B) / s,
        "log_sales":     np.log1p(s),
        "log_assets":    np.log1p(float(assets_B)),
        "Industry":      industry,
        "rev_growth":    rev_growth,
        "ebitda_margin": ebitda_margin,
        "gross_margin":  gross_margin,
    }])


# Display-unit toggle — all values are stored/computed internally in $B
# always (model, scoring, thresholds); this only affects presentation.
UNIT_MULTIPLIER = {"Billions ($B)": 1, "Millions ($M)": 1_000, "Thousands ($K)": 1_000_000}
UNIT_SUFFIX     = {"Billions ($B)": "B", "Millions ($M)": "M", "Thousands ($K)": "K"}


def get_display_unit() -> str:
    return st.session_state.get("display_unit", "Billions ($B)")


def fmt_B(val) -> str:
    if val is None:
        return "—"
    v = float(val)
    unit = get_display_unit()
    if unit == "Billions ($B)" and abs(v) >= 1_000:
        return f"${v/1_000:.2f}T"
    mult   = UNIT_MULTIPLIER[unit]
    suffix = UNIT_SUFFIX[unit]
    v_scaled = v * mult
    if unit == "Billions ($B)":
        return f"${v_scaled:.1f}{suffix}"
    return f"${v_scaled:,.1f}{suffix}"


def fmt_x(val, dec=1) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{float(val):.{dec}f}×"


def fmt_pct(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{float(val)*100:.1f}%"


def thick_divider() -> None:
    """A visually heavier section divider than st.divider(), for marking
    major boundaries in the sidebar (autofill vs. manual input vs. data
    sources, etc.) that deserve more separation than a hairline rule."""
    st.markdown(
        '<hr style="border:none;border-top:4px solid #999;margin:1.1rem 0;">',
        unsafe_allow_html=True,
    )


def big_stat(label: str, value: str, sub: str = "") -> None:
    # Always render the sub line (with a non-breaking space when empty) so all
    # cards have identical height. Single-line HTML avoids Streamlit's markdown
    # parser switching modes on newlines and rendering "<div>" as literal text.
    sub_content = sub if sub else "&nbsp;"
    html = (
        f'<div style="background:#f8f9fa;border-radius:10px;padding:1rem 0.75rem;'
        f'text-align:center;min-height:120px;display:flex;flex-direction:column;justify-content:center">'
        f'<div style="font-size:0.70rem;color:#555;font-weight:600;text-transform:uppercase;letter-spacing:0.7px">{label}</div>'
        f'<div style="font-size:2.1rem;font-weight:800;color:#111;margin-top:5px;line-height:1.1">{value}</div>'
        f'<div style="font-size:0.78rem;color:#888;margin-top:4px;min-height:1.1em">{sub_content}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def verdict_card(label: str, actual, predicted: float) -> None:
    s = VERDICT_STYLE[label]
    display_label = VERDICT_DISPLAY_LABEL.get(label, label)
    if actual is not None:
        # `predicted` here is the blended peer-median/model estimate (see
        # MODEL_WEIGHT/PEER_WEIGHT), not the raw model estimate alone.
        gap = (predicted - actual) / actual * 100
        if label == "UNDERVALUED":
            sub = (f"Market pays <b>{actual:.1f}×</b> revenue &nbsp;·&nbsp; "
                   f"Fair value estimate: <b>{predicted:.1f}×</b> &nbsp;·&nbsp; "
                   f"<b>+{gap:.0f}%</b> potential upside")
        elif label == "OVERVALUED":
            sub = (f"Market pays <b>{actual:.1f}×</b> revenue &nbsp;·&nbsp; "
                   f"Fair value estimate: <b>{predicted:.1f}×</b> &nbsp;·&nbsp; "
                   f"<b>{gap:.0f}%</b> premium to fair value estimate")
        else:
            sub = (f"Market: <b>{actual:.1f}×</b> &nbsp;≈&nbsp; "
                   f"Fair value estimate: <b>{predicted:.1f}×</b> — fairly valued")
    else:
        sub = (f"Estimated fair value: <b>{predicted:.1f}×</b> revenue — "
               f"no market price to compare against (private/unlisted company)")

    label_font_size = "2.4rem" if len(display_label) > 12 else "3.6rem"
    st.markdown(
        f"""<div style="background:{s['bg']};border-radius:16px;padding:2.4rem 1.5rem;
                text-align:center;margin:0.5rem 0 1.2rem">
            <div style="font-size:{label_font_size};font-weight:900;color:{s['text']};
                        letter-spacing:4px;line-height:1">
                {s['emoji']}&nbsp; {display_label}
            </div>
            <div style="font-size:1.15rem;color:{s['text']};margin-top:1rem;
                        opacity:0.9;line-height:1.7">
                {sub}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Valuatorix",
                       page_icon="📊", layout="wide")

    # Pass file mtimes so caches auto-invalidate whenever the pkl or xlsx changes.
    _enriched_path = pathlib.Path("data/forbes_enriched.pkl")
    _xlsx_path     = pathlib.Path("data/forbes_global2000.xlsx")
    _emtime = _enriched_path.stat().st_mtime if _enriched_path.exists() else 0.0
    _xmtime = _xlsx_path.stat().st_mtime     if _xlsx_path.exists()     else 0.0

    df, sector_medians, has_descs, ds_source = load_training_data(_emtime, _xmtime)
    clf, reg, reg_ebitda, _                  = train_models(_emtime, _xmtime)
    tfidf_data                               = build_tfidf_index(_emtime)
    yf_info                                  = load_yf_cache_info()
    industries         = sorted(df["Industry"].dropna().unique())
    all_regions        = sorted(df["region"].dropna().unique())
    live: dict | None  = st.session_state.get("live_data")

    # -----------------------------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.selectbox(
            "Display Units", list(UNIT_MULTIPLIER.keys()),
            key="display_unit",
            help="Applies to manual inputs and all dollar figures on the page.",
        )
        display_unit = get_display_unit()
        unit_mult    = UNIT_MULTIPLIER[display_unit]
        unit_suffix  = UNIT_SUFFIX[display_unit]

        thick_divider()
        st.header("🔍 Search a Company")
        query = st.text_input("Name or ticker",
                              placeholder="e.g. Apple, TSLA, Maersk…")
        selected_symbol = None
        if query and len(query) >= 2:
            rate_limited = False
            try:
                with st.spinner("Searching…"):
                    hits = search_companies(query)
            except YFTemporaryError:
                hits, rate_limited = [], True

            if rate_limited:
                st.warning("Yahoo Finance is rate-limiting requests right now — "
                           "wait a few seconds and try again.")
            elif hits:
                choice = st.selectbox(
                    "Results", ["— select —"] + [h["label"] for h in hits])
                if choice != "— select —":
                    selected_symbol = next(
                        (h["symbol"] for h in hits if h["label"] == choice), None)
            else:
                st.caption("No results — try a different name or ticker.")

        if st.button("Load company", type="primary",
                     use_container_width=True, disabled=(selected_symbol is None)):
            try:
                with st.spinner(f"Loading {selected_symbol}…"):
                    data = fetch_ticker(selected_symbol)
            except YFTemporaryError:
                st.warning("Yahoo Finance is rate-limiting requests right now — "
                           "wait a few seconds and try again.")
            else:
                if data is None:
                    st.error(f"**{selected_symbol}** has no usable data on Yahoo "
                             f"Finance (missing revenue/market cap) — try manual "
                             f"entry below instead.")
                else:
                    st.session_state["live_data"] = data
                    live = data
                    st.success(f"Loaded **{data['name']}**")

        if live:
            if st.button("✕ Clear company", use_container_width=True):
                st.session_state.pop("live_data", None)
                st.rerun()

        thick_divider()
        st.header("✏️ Manual Input")
        st.caption(
            "No ticker or Yahoo Finance connectivity needed — fills in from "
            "whatever is currently set in the fields below. Fill those in "
            "first, or edit and click again after."
        )
        manual_name = st.text_input("Company Name",
                                    placeholder="e.g. Acme Robotics Ltd.")

        _ns = live['ticker'] if live else 'manual'
        if st.button("📥 Load this company", type="primary",
                     use_container_width=True, disabled=(not manual_name.strip())):
            _sel_region_raw = st.session_state.get(f"region_{_ns}", "Other (global)")
            _region = "Other" if _sel_region_raw == "Other (global)" else _sel_region_raw
            _use_extra = st.session_state.get(f"use_extra_{_ns}", True)
            _rev_growth_pct    = st.session_state.get(f"rev_growth_pct_{_ns}", 0.0)
            _ebitda_margin_pct = st.session_state.get(f"ebitda_margin_pct_{_ns}", 15.0)
            _gross_margin_pct  = st.session_state.get(f"gross_margin_pct_{_ns}", 40.0)
            _rev_growth_v    = _rev_growth_pct / 100 if _use_extra else None
            _ebitda_margin_v = _ebitda_margin_pct / 100 if _use_extra else None
            _gross_margin_v  = _gross_margin_pct / 100 if _use_extra else None
            # Financial inputs are keyed by display unit too (see toggle below) —
            # read back through the same unit and convert to canonical $B.
            _sales_B  = st.session_state.get(f"sales_B_{_ns}_{unit_suffix}", 50.0 * unit_mult) / unit_mult
            _profit_B = st.session_state.get(f"profit_B_{_ns}_{unit_suffix}", 5.0 * unit_mult) / unit_mult
            _assets_B = st.session_state.get(f"assets_B_{_ns}_{unit_suffix}", 80.0 * unit_mult) / unit_mult
            manual_data = {
                "name":          manual_name.strip(),
                "ticker":        None,
                "mktcap_B":      None,
                "country":       "",
                "region":        _region,
                "industry":      st.session_state.get(f"industry_{_ns}", "IT Software & Services"),
                "sales_B":       _sales_B,
                "profit_B":      _profit_B,
                "assets_B":      _assets_B,
                "description":   st.session_state.get(f"desc_{_ns}", ""),
                "rev_growth":    _rev_growth_v,
                "ebitda_margin": _ebitda_margin_v,
                "ebitda_B":      _ebitda_margin_v * _sales_B if _ebitda_margin_v is not None else None,
                "gross_margin":  _gross_margin_v,
            }
            st.session_state["live_data"] = manual_data
            live = manual_data
            st.success(f"Loaded **{manual_data['name']}**")

        st.divider()
        st.header(f"📋 Financials  *(in ${unit_suffix})*")
        if live and live.get("ticker"):
            st.caption("Auto-filled from Yahoo Finance — edit any field to override.")
        else:
            st.caption(f"Enter values in US$ {display_unit.split('(')[0].strip()} "
                       f"(e.g. {50*unit_mult:,.0f} = ${50*unit_mult:,.0f}{unit_suffix}).")

        sales_display  = st.number_input(f"Annual Revenue (${unit_suffix})", min_value=0.0,
            value=round(float(live["sales_B"]) * unit_mult, 2) if live else round(50.0 * unit_mult, 2),
            step=0.1, format="%.2f", key=f"sales_B_{_ns}_{unit_suffix}")
        profit_display = st.number_input(f"Net Profit (${unit_suffix})",
            value=round(float(live["profit_B"]) * unit_mult, 2) if live else round(5.0 * unit_mult, 2),
            step=0.1, format="%.2f",
            help="Net income after tax. Can be negative.", key=f"profit_B_{_ns}_{unit_suffix}")
        assets_display = st.number_input(f"Total Assets (${unit_suffix})", min_value=0.0,
            value=round(float(live["assets_B"]) * unit_mult, 2) if live else round(80.0 * unit_mult, 2),
            step=0.1, format="%.2f", key=f"assets_B_{_ns}_{unit_suffix}")
        sales_B  = sales_display / unit_mult
        profit_B = profit_display / unit_mult
        assets_B = assets_display / unit_mult

        default_ind = live["industry"] if live else "IT Software & Services"
        ind_idx     = industries.index(default_ind) if default_ind in industries else 0
        industry    = st.selectbox("Industry", industries, index=ind_idx,
                                   format_func=lambda x: INDUSTRY_LABELS.get(x, x),
                                   key=f"industry_{_ns}")

        st.divider()
        st.header("📈 Growth & Margins  *(optional)*")
        st.caption("Used by the ML model when provided — improves prediction accuracy.")

        def _pct_input(label, live_key, default_val, help_txt, key):
            raw = live.get(live_key) if live else None
            val = round(raw * 100, 2) if raw is not None else default_val
            return st.number_input(label, value=val, step=0.1, format="%.2f", help=help_txt, key=key)

        rev_growth_pct   = _pct_input("Revenue Growth (%)", "rev_growth",   0.0,
                                       "Year-over-year revenue growth, e.g. 12.5 for 12.5%",
                                       key=f"rev_growth_pct_{_ns}")
        ebitda_margin_pct = _pct_input("EBITDA Margin (%)", "ebitda_margin", 15.0,
                                        "EBITDA as % of revenue, e.g. 25 for 25%",
                                        key=f"ebitda_margin_pct_{_ns}")
        gross_margin_pct  = _pct_input("Gross Margin (%)",  "gross_margin",  40.0,
                                        "Gross profit as % of revenue, e.g. 60 for 60%",
                                        key=f"gross_margin_pct_{_ns}")

        use_extra = st.checkbox("Include these in model prediction", value=True,
                                help="Uncheck to fall back to revenue/assets/profit only.",
                                key=f"use_extra_{_ns}")
        rev_growth_v    = rev_growth_pct   / 100 if use_extra else None
        ebitda_margin_v = ebitda_margin_pct / 100 if use_extra else None
        gross_margin_v  = gross_margin_pct  / 100 if use_extra else None

        st.divider()
        st.header("📝 Company Description")
        st.caption(
            "Used for TF-IDF comparable matching (50% weight). "
            "Auto-filled when a company is loaded — edit or type manually."
        )
        desc_key = f"desc_{live['ticker'] if live else 'manual'}"
        default_desc = live.get("description", "") if live else ""

        scrape_url = st.text_input(
            "Or paste a website URL to fetch it from",
            key=f"scrape_url_{live['ticker'] if live else 'manual'}",
            placeholder="https://example.com/about",
        )
        if st.button("🔗 Fetch description from URL", use_container_width=True,
                     disabled=(not scrape_url.strip())):
            with st.spinner("Fetching description…"):
                scraped_text, err = scrape_description_from_url(scrape_url.strip())
            if scraped_text:
                st.session_state[desc_key] = scraped_text
                st.success("Description updated from that page.")
            else:
                st.warning(f"Couldn't fetch a description from that URL — {err}")

        manual_description = st.text_area(
            "What does this company do?",
            value=st.session_state.get(desc_key, default_desc),
            height=120,
            placeholder=(
                "e.g. 'Designs and sells AI chips for data centers and gaming. "
                "Leads the GPU market with CUDA software ecosystem.' "
                "— the more specific, the better the comparable matching."
            ),
            key=desc_key,
        )

        st.divider()
        st.header("🌍 Geography")
        auto_region = live.get("region", "Other") if live else "Other"
        if live and live.get("country"):
            st.caption(f"Yahoo detected: **{live['country']}** → **{auto_region}**  "
                       f"*(change below to override)*")
        region_opts = ["Other (global)"] + all_regions
        region_key  = f"region_{live['ticker'] if live else 'manual'}"
        auto_idx    = region_opts.index(auto_region) if auto_region in region_opts else 0
        sel_region  = st.selectbox("Region for comparable selection",
                                   region_opts, index=auto_idx, key=region_key)
        region = "Other" if sel_region == "Other (global)" else sel_region

        thick_divider()
        st.markdown("**📊 Data sources**")
        if has_descs:
            st.success(f"Descriptions active — TF-IDF enabled ({ds_source})")
        else:
            st.caption(
                f"**ML training + comps pool**  \n"
                f"Forbes Global 2000 · **{len(df):,} companies · 27 industries**  \n"
                f"*Descriptions not yet enriched — run `enrich_forbes_descriptions.py` "
                f"to unlock TF-IDF comparable matching*"
            )
        if yf_info:
            st.caption(
                f"**Yahoo Finance cache**  \n"
                f"{yf_info['n']} companies · fetched {yf_info['date']}  \n"
                f"Used for live company lookups & enrichment only"
            )
        else:
            st.caption(
                "**Yahoo Finance cache:** not found  \n"
                "Run `python scripts/fetch_yf_training.py`  \n"
                "*(optional — only affects live data enrichment)*"
            )

    # -----------------------------------------------------------------------
    # Predictions
    # -----------------------------------------------------------------------
    profit_margin_input = profit_B / max(sales_B, 0.001)
    input_df    = build_input_row(sales_B, assets_B, profit_B, industry,
                                  rev_growth=rev_growth_v,
                                  ebitda_margin=ebitda_margin_v,
                                  gross_margin=gross_margin_v)
    ev_pred     = float(np.expm1(reg.predict(input_df)[0]))   # invert log1p
    ev_pred     = max(ev_pred, 0.0)
    value_prob  = float(clf.predict_proba(input_df)[0][1])

    ev_ebitda_pred = float(np.expm1(reg_ebitda.predict(input_df)[0]))
    ev_ebitda_pred = max(ev_ebitda_pred, 0.0)
    target_ebitda_B  = live.get("ebitda_B") if live else None
    implied_ev_ebitda = ev_ebitda_pred * target_ebitda_B if target_ebitda_B else None
    sector_med  = float(sector_medians.get(industry, df["ev_revenue"].median()))
    ind_label   = INDUSTRY_LABELS.get(industry, industry)

    # manual_description takes priority over YF description (user may have edited it)
    target_desc = manual_description or (live.get("description", "") if live else "")
    comps, rationale = score_comparables(
        df, industry, sales_B, profit_margin_input, region,
        exclude_name=live["name"] if live else "",
        target_description=target_desc,
        tfidf_data=tfidf_data,
        ebitda_margin=ebitda_margin_v)
    comps_med_ev  = comps["ev_revenue"].median()
    comps_mean_ev = comps["ev_revenue"].mean()

    # Consistency fix: for financial companies (banks, insurers) the Forbes pool
    # uses MarketCap as EV (deposits aren't real debt). Use the same convention
    # for the live company's actual multiple so the comparison is apples-to-apples.
    FINANCIAL_SECTORS = {"Banking", "Insurance", "Diversified Financials"}
    if live:
        if industry in FINANCIAL_SECTORS:
            actual_ev_rev = live.get("ps_actual")   # MarketCap/Revenue, same as Forbes pool
        else:
            actual_ev_rev = live.get("ev_revenue")  # proper EV/Revenue
    else:
        actual_ev_rev = None
    blended_benchmark = MODEL_WEIGHT * ev_pred + PEER_WEIGHT * comps_med_ev
    implied_ev        = blended_benchmark * sales_B
    if actual_ev_rev is not None:
        verdict = ("UNDERVALUED" if actual_ev_rev < blended_benchmark * 0.87 else
                   "OVERVALUED"  if actual_ev_rev > blended_benchmark * 1.13 else "FAIR VALUE")
    else:
        # No market price to compare against (private/manual company, or no
        # company loaded) — present an estimate, not an over/undervalued
        # judgment. value_prob is still computed (classifier stays trained
        # and usable) but no longer drives a verdict label here.
        verdict = "ESTIMATE"

    # -----------------------------------------------------------------------
    # A — COMPANY OVERVIEW
    # -----------------------------------------------------------------------
    if live:
        title_col, excel_col = st.columns([4, 1])
        with title_col:
            st.title(f"{live['name']}  ({live['ticker']})" if live.get("ticker") else live["name"])
        excel_button_slot = excel_col.empty()
        if live.get("description"):
            st.markdown(live["description"])

        st.markdown("### Key Financials")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            big_stat("Revenue", fmt_B(live["sales_B"]))
        with c2:
            big_stat(
                "EBITDA",
                fmt_B(live.get("ebitda_B")) if live.get("ebitda_B") else "—",
                sub=(fmt_pct(live.get("ebitda_margin")) + " margin")
                    if live.get("ebitda_margin") else "",
            )
        with c3:
            big_stat("Market Cap", fmt_B(live["mktcap_B"]))
        with c4:
            ev_show = live.get("ev_B", live["mktcap_B"])
            big_stat(
                "Enterprise Value", fmt_B(ev_show),
                sub=f"Net Debt: {fmt_B(live.get('net_debt_B', 0))}",
            )

        st.markdown("")
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            big_stat(
                "Net Profit", fmt_B(live["profit_B"]),
                sub=(fmt_pct(live.get("net_margin")) + " net margin")
                    if live.get("net_margin") else "",
            )
        with c6:
            big_stat("Revenue Growth", fmt_pct(live.get("rev_growth")))
        with c7:
            big_stat("P / E Ratio", fmt_x(live.get("pe_ratio")))
        with c8:
            big_stat("Country", live.get("country") or "—",
                     sub=live.get("region") or "")

        if live.get("yf_industry"):
            st.caption(
                f"Yahoo Finance: **{live['yf_industry']}** "
                f"(sector: {live['yf_sector']}) "
                f"→ Forbes category: **{ind_label}**"
            )
        st.divider()
    else:
        title_col, excel_col = st.columns([4, 1])
        with title_col:
            st.title("📊 Valuatorix — Comparable Companies Valuation")
        excel_button_slot = excel_col.empty()
        st.markdown(
            "Benchmark any company against **1,999 global peers** from the Forbes Global 2000."
        )
        cta_l, cta_r = st.columns(2)
        with cta_l:
            st.info(
                "**🔍 Search for a company**  \n"
                "Type a name or ticker in the sidebar — the app loads live financials "
                "from Yahoo Finance and auto-fills all fields."
            )
        with cta_r:
            st.success(
                "**✏️ Manually fill in the fields on the left**  \n"
                "Enter Revenue, Profit, Assets, Industry — and optionally a business "
                "description to unlock description-based comparable matching.  \n"
                "*Works for any company, including private ones not on Yahoo Finance.*"
            )
        st.markdown(
            "Either way, the model will estimate a fair EV/Revenue multiple and "
            "find the closest comparable companies. For public companies, it shows "
            "whether the stock appears undervalued, overvalued, or fairly priced "
            "against that estimate; for private/manual entries with no market price "
            "to compare against, it shows the estimated valuation instead."
        )
        st.divider()

    # Feature-driver ranking, computed here so it's available inside the
    # Model Estimate column below.
    medians = industry_feature_medians(_emtime, _xmtime)
    drivers = explain_verdict(reg, input_df, industry, medians)

    def _fmt_driver(feat, val):
        return fmt_B(val) if feat in ("Sales_B", "Assets_B", "Profits_B") else fmt_pct(val)

    # -----------------------------------------------------------------------
    # B — VALUATION
    # -----------------------------------------------------------------------
    st.markdown("## Valuation")

    # Quick overview: market EV vs. the blended intrinsic estimate, side by
    # side, so market and model/peer views are directly comparable at a
    # glance. Only shown when there's a real market to compare against —
    # manual/private companies show a single estimate box instead (see
    # Verdict section below for why no over/undervalued label applies then).
    if live and actual_ev_rev:
        qo1, qo2 = st.columns(2)
        with qo1:
            big_stat(
                "Current Market EV",
                fmt_B(live.get("ev_B", live["mktcap_B"])),
                sub=f"{actual_ev_rev:.1f}× EV/Revenue (market)",
            )
        with qo2:
            big_stat(
                "Model + Peer Estimate",
                fmt_B(implied_ev),
                sub=(f"{blended_benchmark:.1f}× EV/Revenue "
                     f"({MODEL_WEIGHT*100:.0f}% model / {PEER_WEIGHT*100:.0f}% peer median)"),
            )
        st.markdown("")
    else:
        big_stat(
            "Estimated Enterprise Value",
            fmt_B(implied_ev),
            sub=(f"{blended_benchmark:.1f}× EV/Revenue "
                 f"({MODEL_WEIGHT*100:.0f}% model / {PEER_WEIGHT*100:.0f}% peer median) "
                 "— no market price to compare against"),
        )
        st.markdown("")

    col_mkt, col_mdl = st.columns(2)

    with col_mkt:
        st.markdown("#### 📌 Current Market Price")
        if live and actual_ev_rev:
            for lbl, val in [
                ("EV / Revenue",           fmt_x(live.get("ev_revenue"))),
                ("EV / EBITDA",            fmt_x(live.get("ev_ebitda"))),
                ("Price / Earnings (P/E)", fmt_x(live.get("pe_ratio"))),
                ("Price / Sales  (P/S)",   fmt_x(live.get("ps_actual"))),
                ("Price / Book  (P/B)",    fmt_x(live.get("pb_ratio"))),
            ]:
                lc, vc = st.columns([2, 1])
                lc.markdown(f"**{lbl}**")
                vc.markdown(f"`{val}`")
            st.caption(
                f"EV = Mkt Cap {fmt_B(live['mktcap_B'])} "
                f"+ Net Debt {fmt_B(live.get('net_debt_B', 0))} "
                f"= **{fmt_B(live.get('ev_B', live['mktcap_B']))}**"
            )
        else:
            st.markdown(
                f"Revenue: **{fmt_B(sales_B)}**  \n"
                f"Sector median EV/Revenue: **{sector_med:.1f}×**  \n"
                f"*Search a company for full multiples.*"
            )

    with col_mdl:
        st.markdown("#### 🤖 Model Estimate  *(EV / Revenue)*")
        for lbl, val in [
            (f"Model Prediction ({MODEL_WEIGHT*100:.0f}% weight)", fmt_x(ev_pred)),
            (f"Peer Median ({PEER_WEIGHT*100:.0f}% weight)",       fmt_x(comps_med_ev)),
            ("Peer Mean EV/Revenue",     fmt_x(comps_mean_ev)),
            ("→ Fair Value Estimate",    fmt_x(blended_benchmark)),
            ("Implied Enterprise Value", fmt_B(implied_ev)),
        ]:
            lc, vc = st.columns([2, 1])
            lc.markdown(f"**{lbl}**")
            vc.markdown(f"`{val}`")
        st.caption(
            f"Trained on **{len(df):,} Forbes companies**. "
            f"Peer pool: **{len(comps)}** comparable {ind_label} companies."
        )
        if target_ebitda_B:
            st.markdown("**Model Predicted EV/EBITDA**")
            lc2, vc2 = st.columns([2, 1])
            lc2.markdown("`Implied EV (via EBITDA)`")
            vc2.markdown(f"`{fmt_x(ev_ebitda_pred)}` → `{fmt_B(implied_ev_ebitda)}`")
            st.caption(
                "Trained on the ~48% of companies with EBITDA data — "
                "treat as lower-confidence than the EV/Revenue estimate above."
            )
        if drivers:
            bits = [
                f"**{d['label']}** ({_fmt_driver(d['feature'], d['actual'])} vs. industry median "
                f"{_fmt_driver(d['feature'], d['median'])})"
                for d in drivers[:2]
            ]
            st.markdown(f"The model's estimate is driven mainly by {' and '.join(bits)}.")

    st.divider()

    # -----------------------------------------------------------------------
    # C — VERDICT
    # -----------------------------------------------------------------------
    verdict_card(verdict, actual_ev_rev, blended_benchmark)

    if actual_ev_rev:
        gap_pct = (blended_benchmark - actual_ev_rev) / actual_ev_rev * 100
        blend_note = (f"({MODEL_WEIGHT*100:.0f}% model **{ev_pred:.1f}×** "
                      f"/ {PEER_WEIGHT*100:.0f}% peer median **{comps_med_ev:.1f}×**)")
        if verdict == "UNDERVALUED":
            st.info(
                f"**{live['name']}** trades at **{actual_ev_rev:.1f}×** EV/Revenue. "
                f"Fair value estimate: **{blended_benchmark:.1f}×** {blend_note}. "
                f"The stock appears **{abs(gap_pct):.0f}% below** that fair value estimate."
            )
        elif verdict == "OVERVALUED":
            st.warning(
                f"**{live['name']}** trades at **{actual_ev_rev:.1f}×** EV/Revenue. "
                f"Fair value estimate: **{blended_benchmark:.1f}×** {blend_note}. "
                f"The stock trades at a **{abs(gap_pct):.0f}% premium** to that fair value estimate."
            )
        else:
            st.success(
                f"**{live['name']}** at **{actual_ev_rev:.1f}×** EV/Revenue is in line with the "
                f"fair value estimate of **{blended_benchmark:.1f}×** {blend_note}."
            )
    else:
        st.info(
            f"Based on **{len(comps)} comparable {ind_label} companies**, "
            f"the estimated fair value is **{blended_benchmark:.1f}×** EV/Revenue "
            f"({MODEL_WEIGHT*100:.0f}% model **{ev_pred:.1f}×** / {PEER_WEIGHT*100:.0f}% peer median "
            f"**{comps_med_ev:.1f}×**) → implied enterprise value **{fmt_B(implied_ev)}**. "
            f"Sector median: **{sector_med:.1f}×**. "
            f"No market price exists for this company, so this is presented as an "
            f"estimate rather than an undervalued/overvalued judgment."
        )

    if live and live.get("ticker"):
        with st.spinner("Loading price history…"):
            hist = fetch_price_history(live["ticker"])
        pct_ch = None
        if hist is not None and not hist.empty:
            start_p = hist["Close"].iloc[0]
            end_p   = hist["Close"].iloc[-1]
            pct_ch  = (end_p - start_p) / start_p * 100
        news = fetch_company_news(live["ticker"])

        if pct_ch is not None or news:
            st.markdown("**Market cross-check**")
            if pct_ch is not None:
                trend_line = f"1-year price change: **{pct_ch:+.1f}%**"
                if verdict in ("UNDERVALUED", "OVERVALUED") and pct_ch != 0:
                    rising = pct_ch > 0
                    # OVERVALUED: rising = consistent with paying a premium.
                    # UNDERVALUED: falling = consistent with an ongoing discount.
                    consistent = rising if verdict == "OVERVALUED" else not rising
                    trend_line += (
                        " — consistent with the verdict." if consistent
                        else " — doesn't obviously support the verdict."
                    )
                st.markdown(trend_line)
            if news:
                st.markdown("**Recent news**")
                for n in news:
                    headline = f"[{n['title']}]({n['url']})" if n.get("url") else n["title"]
                    date = n["pubDate"][:10] if n.get("pubDate") else ""
                    meta_parts = [p for p in (n.get("publisher"), date) if p]
                    suffix = f"  \n  *{' — '.join(meta_parts)}*" if meta_parts else ""
                    st.markdown(f"- {headline}{suffix}")

    st.divider()

    # -----------------------------------------------------------------------
    # D — COMPARABLE COMPANIES
    # -----------------------------------------------------------------------
    st.markdown(f"## Comparable Companies — {ind_label}")
    ev_note = ("EV = Market Cap + Net Debt" if "net_debt_B" in df.columns
               else "EV ≈ Market Cap (net debt not yet enriched)")
    desc_active = bool(target_desc and len(target_desc) > 80 and has_descs)
    if desc_active:
        scoring_note = "Scoring: description 50% · industry 20% · geography 15% · revenue 10% · margin 5%"
    else:
        scoring_note = "Scoring: industry 40% · geography 30% · revenue 20% · margin 10% (add a description to unlock 50% description-similarity weight)"
    st.caption(
        f"**{len(comps)} peers** selected from Forbes Global 2000 (1,999-company pool) · "
        f"{scoring_note} · "
        f"Peer median EV/Revenue: **{comps_med_ev:.2f}×** · "
        f"Range: {comps['ev_revenue'].min():.2f}× – {comps['ev_revenue'].max():.2f}× · "
        f"*{ev_note}*"
    )

    # Build full display table
    display_cols = ["Company", "Country", "Sales_B", "Profits_B",
                    "profit_margin", "ev_B", "ev_revenue", "ev_ebitda", "_score"]
    # ev_B may not exist if no net debt data — fall back to MarketValue_B
    if "ev_B" not in comps.columns:
        comps = comps.copy()
        comps["ev_B"] = comps["MarketValue_B"]
    cd = comps[display_cols].copy().rename(columns={
        "Sales_B":       "Revenue $B",
        "Profits_B":     "Net Profit $B",
        "profit_margin": "Margin %",
        "ev_B":          "EV $B",
        "ev_revenue":    "EV/Rev ×",
        "ev_ebitda":     "EV/EBITDA ×",
        "_score":        "Comp Score",
    })
    cd["Revenue $B"]    = cd["Revenue $B"].round(1)
    cd["Net Profit $B"] = cd["Net Profit $B"].round(1)
    cd["Margin %"]      = (cd["Margin %"] * 100).round(1).astype(str) + "%"
    cd["EV $B"]         = cd["EV $B"].round(1)
    cd["EV/Rev ×"]      = cd["EV/Rev ×"].round(2)
    cd["EV/EBITDA ×"]   = cd["EV/EBITDA ×"].apply(
        lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    cd["Comp Score"]    = cd["Comp Score"].astype(int).astype(str) + " / 100"

    comps_ebitda_valid = comps["ev_ebitda"].dropna()
    median_ev_ebitda = f"{comps_ebitda_valid.median():.2f}" if len(comps_ebitda_valid) else "—"

    median_row = pd.DataFrame([{
        "Company": "── Peer Median ──", "Country": "",
        "Revenue $B":    round(comps["Sales_B"].median(), 1),
        "Net Profit $B": round(comps["Profits_B"].median(), 1),
        "Margin %":      f"{comps['profit_margin'].median()*100:.1f}%",
        "EV $B":         round(comps["ev_B"].median(), 1),
        "EV/Rev ×":      round(comps_med_ev, 2),
        "EV/EBITDA ×":   median_ev_ebitda,
        "Comp Score":    "",
    }])
    mean_row = pd.DataFrame([{
        "Company": "── Peer Mean ──", "Country": "",
        "Revenue $B":    round(comps["Sales_B"].mean(), 1),
        "Net Profit $B": round(comps["Profits_B"].mean(), 1),
        "Margin %":      f"{comps['profit_margin'].mean()*100:.1f}%",
        "EV $B":         round(comps["ev_B"].mean(), 1),
        "EV/Rev ×":      round(comps_mean_ev, 2),
        "EV/EBITDA ×":   (f"{comps_ebitda_valid.mean():.2f}" if len(comps_ebitda_valid) else "—"),
        "Comp Score":    "",
    }])
    if live and actual_ev_rev:
        target_ev_ebitda = live.get("ev_ebitda")
        target_row = pd.DataFrame([{
            "Company":       f"▶ {live['name']} (TARGET)",
            "Country":       live.get("country", ""),
            "Revenue $B":    round(live["sales_B"], 1),
            "Net Profit $B": round(live["profit_B"], 1),
            "Margin %":      (f"{live['net_margin']*100:.1f}%"
                              if live.get("net_margin") else "—"),
            "EV $B":         round(live.get("ev_B", live["mktcap_B"]), 1),
            "EV/Rev ×":      round(actual_ev_rev, 2),
            "EV/EBITDA ×":   (f"{target_ev_ebitda:.2f}" if target_ev_ebitda else "—"),
            "Comp Score":    "TARGET",
        }])
        full_table = pd.concat([cd, median_row, mean_row, target_row], ignore_index=True)
    else:
        full_table = pd.concat([cd, median_row, mean_row], ignore_index=True)

    # full_table itself stays canonical $B (used for CSV/Excel exports, so
    # downloads are always unambiguous regardless of the on-screen toggle).
    # full_table_display is a unit-scaled copy for the on-screen table only.
    full_table_display = full_table.copy()
    for _col in ["Revenue $B", "Net Profit $B", "EV $B"]:
        full_table_display[_col] = pd.to_numeric(full_table_display[_col], errors="coerce") * unit_mult
    full_table_display = full_table_display.rename(columns={
        "Revenue $B":    f"Revenue ${unit_suffix}",
        "Net Profit $B": f"Net Profit ${unit_suffix}",
        "EV $B":         f"EV ${unit_suffix}",
    })

    # Download buttons — CSV stays here; Excel renders into the top-of-page
    # placeholder declared next to the title (see excel_button_slot), per R9.
    dl_name   = f"comps_{industry.replace(' & ','_').replace(' ','_').lower()}.csv"
    col_dl, col_sp = st.columns([1, 4])
    with col_dl:
        st.download_button(
            label="⬇️ Download CSV",
            data=full_table.to_csv(index=False).encode(),
            file_name=dl_name,
            mime="text/csv",
            use_container_width=True,
        )
    # Excel data-gathering stays here (after drivers/full_table are ready);
    # only the button's render target is the top-of-page placeholder, per the
    # KTD — this avoids hoisting the whole comps/drivers dependency chain.
    summary_rows = [
        {"Field": "Company",  "Value": live["name"] if live else "(manual entry, no company loaded)"},
    ]
    if live and live.get("ticker"):
        summary_rows.append({"Field": "Ticker", "Value": live["ticker"]})
    summary_rows += [
        {"Field": "Industry",                "Value": ind_label},
        {"Field": "Region",                  "Value": live.get("region", "") if live else region},
        {"Field": "Revenue ($B)",            "Value": round(sales_B, 3)},
        {"Field": "Net Profit ($B)",         "Value": round(profit_B, 3)},
        {"Field": "Total Assets ($B)",       "Value": round(assets_B, 3)},
        {"Field": "Verdict",                 "Value": VERDICT_DISPLAY_LABEL.get(verdict, verdict)},
        {"Field": f"Model Prediction ({MODEL_WEIGHT*100:.0f}% weight)", "Value": fmt_x(ev_pred)},
        {"Field": f"Peer Median ({PEER_WEIGHT*100:.0f}% weight)",       "Value": fmt_x(comps_med_ev)},
        {"Field": "Fair Value Estimate (blend)", "Value": fmt_x(blended_benchmark)},
        {"Field": "Implied Enterprise Value","Value": fmt_B(implied_ev)},
        {"Field": "Peer Mean EV/Revenue",    "Value": fmt_x(comps_mean_ev)},
        {"Field": "Sector Median EV/Revenue","Value": fmt_x(sector_med)},
    ]
    if actual_ev_rev:
        summary_rows.append({"Field": "Actual EV/Revenue", "Value": fmt_x(actual_ev_rev)})
    if target_ebitda_B:
        summary_rows.append({"Field": "Model Predicted EV/EBITDA", "Value": fmt_x(ev_ebitda_pred)})
    if drivers:
        bits = [
            f"{d['label']} ({_fmt_driver(d['feature'], d['actual'])} vs. industry median "
            f"{_fmt_driver(d['feature'], d['median'])})"
            for d in drivers[:2]
        ]
        summary_rows.append({
            "Field": "Why this verdict",
            "Value": f"The model's estimate is driven mainly by {' and '.join(bits)}.",
        })
    # Market cross-check is recomputed independently here, guarded the same
    # way the on-screen block is — those locals are scoped inside that
    # block and not safely referenceable from this later code path.
    if live and live.get("ticker"):
        xl_hist = fetch_price_history(live["ticker"])
        if xl_hist is not None and not xl_hist.empty:
            xl_start = xl_hist["Close"].iloc[0]
            xl_end   = xl_hist["Close"].iloc[-1]
            xl_pct_ch = (xl_end - xl_start) / xl_start * 100
            summary_rows.append({"Field": "1-Year Price Change", "Value": f"{xl_pct_ch:+.1f}%"})
        xl_news = fetch_company_news(live["ticker"])
        for i, n in enumerate(xl_news[:5], start=1):
            headline = f"{n['title']} ({n['url']})" if n.get("url") else n["title"]
            summary_rows.append({"Field": f"News {i}", "Value": headline})

    xl_buf = io.BytesIO()
    with pd.ExcelWriter(xl_buf, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        full_table.to_excel(writer, sheet_name="Comparable Companies", index=False)

    excel_button_slot.download_button(
        label="📊 Download Full Report (Excel)",
        data=xl_buf.getvalue(),
        file_name=f"report_{industry.replace(' & ','_').replace(' ','_').lower()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.dataframe(full_table_display, use_container_width=True, height=640, hide_index=True)

    # Bubble chart: Revenue vs EV/Revenue, bubble size = Enterprise Value, coloured by region
    region_colors = {
        "North America":      "#4C72B0",
        "Europe":             "#DD8452",
        "Asia-Pacific":       "#55A868",
        "Middle East & Africa": "#C44E52",
        "Latin America":      "#8172B3",
        "Other":              "#BBBBBB",
    }
    ev_for_size  = comps["ev_B"].clip(lower=0.01)
    sizeref      = 2.0 * ev_for_size.max() / (44.0 ** 2)
    fig_sc = go.Figure()
    for reg, grp in comps.groupby("region"):
        grp_size = grp["ev_B"].clip(lower=0.01)  # bubble sizing stays in $B, independent of display unit
        grp_cd = grp[["Company", "Country", "profit_margin", "ev_B", "_score"]].copy()
        grp_cd["ev_B"] = grp_cd["ev_B"] * unit_mult
        fig_sc.add_trace(go.Scatter(
            x=grp["Sales_B"] * unit_mult, y=grp["ev_revenue"], mode="markers",
            name=reg,
            marker=dict(
                size=grp_size, sizemode="area", sizeref=sizeref, sizemin=4,
                color=region_colors.get(reg, "#aaa"),
                line=dict(width=1, color="white"), opacity=0.85,
            ),
            customdata=grp_cd.values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                f"Revenue: $%{{x:.1f}}{unit_suffix} · EV/Rev: %{{y:.2f}}×<br>"
                f"EV: $%{{customdata[3]:.1f}}{unit_suffix} · Margin: %{{customdata[2]:.1%}} · "
                "Comp score: %{customdata[4]:.0f}/100<extra></extra>"
            ),
        ))
    fig_sc.add_hline(y=ev_pred, line_dash="dash", line_color="green",
                     annotation_text=f"Model: {ev_pred:.1f}×",
                     annotation_font_size=13)
    fig_sc.add_hline(y=comps_med_ev, line_dash="dot", line_color="#555",
                     annotation_text=f"Peer median: {comps_med_ev:.1f}×",
                     annotation_font_size=13)
    if actual_ev_rev:
        target_ev_size = max(live.get("ev_B", live["mktcap_B"]), 0.01)  # sizing stays in $B
        fig_sc.add_trace(go.Scatter(
            x=[sales_B * unit_mult], y=[actual_ev_rev], mode="markers+text",
            marker=dict(
                size=[target_ev_size], sizemode="area", sizeref=sizeref, sizemin=14,
                color="red", symbol="star", line=dict(width=1.5, color="#7a0000"),
            ),
            text=[live["name"]], textposition="top right",
            customdata=[[live["name"], live.get("country", ""), target_ev_size * unit_mult]],
            hovertemplate=(
                "<b>%{customdata[0]} (target)</b><br>%{customdata[1]}<br>"
                f"Revenue: $%{{x:.1f}}{unit_suffix} · EV/Rev: %{{y:.2f}}× · EV: $%{{customdata[2]:.1f}}{unit_suffix}"
                "<extra></extra>"
            ),
            showlegend=True, name=f"▶ {live['name']} (target)",
        ))
    fig_sc.update_layout(
        height=440, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title=f"Revenue (${unit_suffix})", yaxis_title="EV / Revenue (×)",
        legend=dict(orientation="h", y=-0.22),
        plot_bgcolor="white",
    )
    fig_sc.update_xaxes(gridcolor="#eee")
    fig_sc.update_yaxes(gridcolor="#eee")
    st.caption("Bubble size = Enterprise Value")
    st.plotly_chart(fig_sc, use_container_width=True)

    with st.expander("📐 How were these comparables selected? (click to expand)"):
        for line in rationale:
            st.markdown(f"- {line}")
        st.markdown(
            "\n> **Data note:** Forbes Global 2000 includes revenue, profit, assets, and "
            "market value. EV/Revenue = (Market Cap + Net Debt) / Revenue. "
            "When the enriched dataset is not yet available, Market Cap is used as EV "
            "(net debt assumed ≈ 0). Run `python scripts/enrich_forbes_descriptions.py` "
            "to add net debt data and enable proper EV/Revenue multiples."
        )

    st.divider()

    # -----------------------------------------------------------------------
    # E — PRICE CHART
    # -----------------------------------------------------------------------
    if live and live.get("ticker"):
        st.markdown(f"## {live['name']} — 1-Year Stock Price")
        with st.spinner(f"Loading price history…"):
            hist = fetch_price_history(live["ticker"])
        if hist is not None:
            start_p = hist["Close"].iloc[0]
            end_p   = hist["Close"].iloc[-1]
            pct_ch  = (end_p - start_p) / start_p * 100
            clr     = "#27ae60" if pct_ch >= 0 else "#e74c3c"
            st.markdown(
                f"<span style='font-size:1.6rem;font-weight:800;color:{clr}'>"
                f"{'+'if pct_ch>=0 else ''}{pct_ch:.1f}%</span>"
                f"&nbsp;&nbsp;"
                f"<span style='font-size:1rem;color:#666'>"
                f"${start_p:.2f} → ${end_p:.2f} over 12 months</span>",
                unsafe_allow_html=True,
            )
            rgba  = f"rgba({'39,174,96' if pct_ch>=0 else '231,76,60'}, 0.12)"
            fig_p = go.Figure(go.Scatter(
                x=hist["Date"], y=hist["Close"], mode="lines",
                line=dict(color=clr, width=2.5),
                fill="tozeroy", fillcolor=rgba,
                hovertemplate="<b>%{x|%b %d %Y}</b>  $%{y:.2f}<extra></extra>",
            ))
            fig_p.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="", yaxis_title="Price (USD)",
                showlegend=False, hovermode="x unified",
            )
            st.plotly_chart(fig_p, use_container_width=True)
        st.divider()

    # -----------------------------------------------------------------------
    # F — SECTOR OVERVIEW
    # -----------------------------------------------------------------------
    st.markdown("## All Sectors — Median EV/Revenue Multiple")
    ev_note_bar = ("EV = Market Cap + Net Debt" if "net_debt_B" in df.columns
                   else "EV ≈ Market Cap (run enrich_forbes_descriptions.py to add net debt)")
    st.caption(
        f"Each bar = median EV/Revenue for all companies in that sector in Forbes Global 2000 · "
        f"Red bar = currently selected industry, all others blue · {ev_note_bar}"
    )
    sps = (
        df.groupby("Industry").agg(
            **{"Median EV/Rev": ("ev_revenue", "median"), "n_companies": ("ev_revenue", "size")}
        )
        .reset_index()
        .sort_values("Median EV/Rev", ascending=True)
    )
    sps["Label"] = sps["Industry"].map(lambda x: INDUSTRY_LABELS.get(x, x))
    fig_bar = go.Figure(go.Bar(
        x=sps["Median EV/Rev"], y=sps["Label"], orientation="h",
        marker=dict(
            color=["#C44E52" if s == industry else "#4C72B0" for s in sps["Industry"]],
        ),
        customdata=sps[["n_companies"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>Median EV/Rev: %{x:.2f}×<br>"
            "Companies: %{customdata[0]}<extra></extra>"
        ),
    ))
    fig_bar.add_vline(x=ev_pred, line_dash="dash", line_color="green",
                      annotation_text=f"Model: {ev_pred:.1f}×",
                      annotation_font_size=13)
    if actual_ev_rev:
        fig_bar.add_vline(x=actual_ev_rev, line_dash="dot", line_color="orange",
                          annotation_text=f"Actual: {actual_ev_rev:.1f}×",
                          annotation_font_size=13)
    fig_bar.update_layout(
        height=600, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Median EV/Revenue (×)", plot_bgcolor="white",
    )
    fig_bar.update_xaxes(gridcolor="#eee")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.caption(
        f"Training: Forbes Global 2000 (2026) · {len(df):,} companies · "
        "EV/Revenue = Enterprise Value ÷ Revenue · For informational purposes only — not financial advice."
    )


if __name__ == "__main__":
    main()
