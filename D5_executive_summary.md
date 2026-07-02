# D5 — Executive Summary

**Project:** Valuatorix — Comparable Companies Valuation Tool
**Author:** René Mohoric
**Date:** July 2026

---

## The Question

Investment banks and private equity firms value acquisition targets using **comparable companies** — public peers with similar financials, sector exposure, and business models. Building a good comps table is time-consuming: it requires sourcing a large peer universe, filtering by industry, and scoring each peer on multiple dimensions. This project automates that process.

Using Forbes Global 2000 (1,901 public companies across 27 industries, after removing EV/Revenue outliers), three ML tasks are addressed:

1. **EV/Revenue multiple prediction:** Given a company's revenue, profit, assets, margins, and sector — what Enterprise Value / Revenue multiple does the market assign? This is the primary M&A valuation multiple for companies without reliable EBITDA (high-growth, early-stage, or capital-intensive firms).
2. **EV/EBITDA multiple prediction (secondary):** The same question using EBITDA instead of revenue as the denominator — often the more appropriate multiple for capital-intensive businesses, trained on the roughly one-third of companies with usable EBITDA data.
3. **Undervaluation screening:** For public companies, is the stock trading *below* a blended fair-value estimate? Companies at a discount are potential value opportunities or acquisition targets trading cheap relative to peers.

---

## What the Tool Does

**Input — any of three ways:**

- Search by ticker (AAPL, DBK.DE, 7203.T) — live data loads from Yahoo Finance, with currency figures automatically converted to USD
- Search by company name — auto-matched to the closest YF ticker
- Manual entry — Revenue, Net Profit, Total Assets, Industry, plus an optional business description or a pasted company-website URL; click "Load this company" to snapshot it the same way a ticker load works. Works for any company, including private firms not on Yahoo Finance

**Processing:**

- Two GradientBoosting regressors (trained on 1,901 Forbes companies) predict fair EV/Revenue (CV R² = 0.45) and EV/EBITDA multiples from the company's fundamentals
- The comparable companies table finds the 20 closest peers by a 5-factor composite similarity score:
  - *With a description:* 50% description similarity (TF-IDF), 20% industry classification, 15% geography, 10% revenue size, 5% margin similarity
  - *Without a description:* 40% industry classification, 30% geography, 20% revenue size, 10% margin similarity
  - The peer pool includes related industries (e.g. Semiconductors for a Software target), not just an exact industry match — related-industry peers are scored down rather than excluded
- For financial companies (banks, insurers): EV = Market Cap only — customer deposits are excluded from net debt, consistent with market convention

**Output:**

- Predicted fair EV/Revenue and EV/EBITDA multiples, with implied enterprise value
- For public companies: a verdict (Undervalued / Overvalued / Fair Value), based on a blend of **30% the model's own estimate and 70% the peer median** — not the model alone
- For private/manually-entered companies: no undervalued/overvalued judgment (there's no market price to compare against) — a plain **Estimated Valuation** figure instead
- A market-EV-vs-model-estimate quick-comparison view for public companies, side by side
- Table of 20 closest comparable companies with EV/Revenue, EV/EBITDA multiples and financials
- Scatter chart: peer universe with the target company highlighted
- 1-year stock price chart (for public companies loaded via ticker)
- A toggle to display every dollar figure in Billions, Millions, or Thousands
- One-click download: comps table as CSV, or the full report as a multi-sheet Excel workbook

---

## What I Found

**Model performance:** GradientBoosting on log1p(EV/Revenue) achieves CV R² = 0.45 — reasonable for a right-skewed financial multiple with wide natural dispersion. The key predictors, in order of importance:

1. **Industry** — the single largest driver. Software companies trade at 5–15× EV/Revenue; banks and utilities at 1–3×.
2. **Profit margin** — within any industry, higher-margin companies command premium multiples.
3. **Revenue scale** — larger companies trade at modest discounts as growth slows (log-linear relationship).
4. **Revenue growth, EBITDA margin, gross margin** — enrichment features with ~47–48% coverage in the Forbes pool; improve predictions for companies where these data points are available.

**Feature selection matters, and doesn't cost accuracy.** The companion analysis (`analysis.qmd`) applies Recursive Feature Elimination inside a leakage-safe pipeline, cutting the transformed feature set in half — validation performance holds up on the reduced set, and the eliminated features are mostly raw size columns (`Sales_B`, `Assets_B`) made redundant by their log-transformed counterparts.

**Business description matching** substantially improves comparable selection. Within broad Forbes categories like "IT Software & Services," the spread in EV/Revenue runs from 0.1× (IT distributors like Ingram Micro) to 50×+ (SaaS / AI-native companies). TF-IDF on business descriptions distinguishes these sub-segments in ways that industry labels alone cannot — it's now the single largest factor in comparable-company scoring (50% weight).

**Financial sector treatment matters.** Banks like Deutsche Bank report customer deposits as "total debt" in Yahoo Finance, producing negative Enterprise Values and nonsensical EV/Revenue multiples. The tool sets net debt = 0 for Banking, Insurance, and Diversified Financials — aligning with how practitioners treat these sectors.

**A currency bug in the raw data was found and fixed.** Non-USD-reporting companies (Japanese, Korean, and other tickers) were showing wildly wrong dollar figures — Toyota's revenue, for example, initially computed at roughly $50 trillion instead of its real ~$314 billion — because Yahoo Finance reports these fields in the company's local currency, not USD. This wasn't just a display bug: it had already corrupted `net_debt_B` for ~170 companies in the training data itself, distorting the enterprise-value figures the regression model trains on. Fixing it required converting every dollar-denominated field to USD using the correct exchange rate before any calculation — and re-running the correction against the historical dataset, not just the live lookup path.

---

## What I Would Recommend

**For M&A analysts and investors:**

Use the regression estimate as a first-pass valuation anchor before building a detailed comps model. Enter the target's revenue, profit margin, assets, and industry — the model returns an implied EV/Revenue multiple and enterprise value. If you have a business description, paste it (or paste a company website URL and let the app pull the text) to unlock description-based comparable selection, which will substantially sharpen the peer group.

For public companies, use the undervalued/overvalued verdict to screen the Forbes G2000 universe: stocks trading well below the blended model/peer estimate may represent mispriced or overlooked assets. For private companies, treat the estimated valuation as a directional anchor, not a verdict — there's no market price behind it to validate against.

**Limitations:**

- EV/Revenue is the right multiple for growth companies but not for capital-intensive businesses (real estate, utilities) where EV/EBITDA is more appropriate. The tool now trains a dedicated EV/EBITDA model, shown as a supplementary metric — but it's built on a smaller, noisier subset of the data (~34% EBITDA coverage) and should be trusted less than the primary EV/Revenue estimate.
- CV R² of 0.45 (EV/Revenue) is useful for directional analysis, not M&A bid precision. Treat the output as a floor/sanity check, not a final number.
- The model generalises across Forbes-scale companies ($1B+ revenue). It should not be applied to SMEs or venture-stage businesses.
- Financial metrics (EBITDA margin, gross margin, revenue growth) are currently available for ~47–48% of the Forbes pool. Coverage improves as Yahoo Finance data is enriched.
- The app has not yet been deployed to a public URL — it currently runs locally only.

---

## How to Access

- **Live app:** `streamlit run app.py` (see README for setup) — not yet deployed to a public URL
- **Companion analysis:** `analysis.qmd` — Forbes Global 2000 EV/Revenue model, full pipeline (renders with Quarto)
- **Slide deck:** `presentation.qmd` — Forbes Global 2000 EV/Revenue presentation (renders with Quarto)
