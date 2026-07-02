# D3 — AI Workflow Reflection

**Author:** René Mohoric
**Tool used:** Claude Code (Claude Sonnet 5) via the CLI and desktop app

---

## Which AI tools I used

**Claude Code (primary)** — the Anthropic CLI tool that runs Claude directly in the terminal alongside the project files. I used it throughout the project for code generation, debugging, and document writing.

**MCP servers (Model Context Protocol):**
- `playwright` — used throughout for browser-based UI verification (clicking through the sidebar, loading companies, checking rendered output). Proved unreliable for state-heavy checks (stale page state after some navigations); for those specific cases I switched to Streamlit's own `AppTest` framework, which drives the app programmatically without a browser and proved more reliable for confirming widget/session-state behavior
- `mcp-mermaid` and `claude-mermaid` — registered to generate the architecture diagram PNG for the interim assignment
- `claude.ai/Gamma` — used to generate a presentation draft

**Claude Code skills (compound-engineering plugin), used repeatedly across the project's lifetime:**
- `/ce-brainstorm` — used for early dataset/scope pivots (AJPES/Slovenian EBITDA, later abandoned; the Forbes Global 2000 approach) and later for smaller feature-shaped decisions (e.g. how a new "why this verdict" explanation should be placed and worded) before planning them
- `/ce-plan` — used repeatedly to turn feature requests into reviewed implementation plans, including a multi-persona document-review pass (coherence, feasibility, scope, design, adversarial reviewers) before implementation on anything touching more than a couple of files
- `/ce-work` — used to execute each plan and generate the corresponding code

---

## How the project evolved

The project went through several pivots and, after the working tool existed, a long second phase of iterative refinement driven by direct usage:

**Phase 1 — AJPES / Capital IQ (abandoned):** The original idea was to train on real M&A deal data from Capital IQ. This was abandoned when it became clear that Capital IQ data cannot go into a public GitHub repository (licence restriction), and the AJPES Slovenian company register data was too regional to be meaningful for a general M&A tool.

**Phase 2 — Fortune 500 baseline:** Pivoted to the publicly available Fortune 500 dataset (2023, 476 public companies). Built a model ladder (Dummy → Linear → Decision Tree → XGBoost) predicting the P/S (Price/Sales) ratio.

**Phase 3 — Forbes Global 2000 live tool:** Rebuilt the Streamlit app from scratch on Forbes Global 2000 (1,999 companies with usable revenue data, 27 industries), with three major improvements:
- EV/Revenue instead of P/S (adds net debt, more accurate for leveraged or cash-rich companies)
- Live Yahoo Finance lookups for any public company by ticker or name
- Description-based TF-IDF comparable matching — distinguishes SaaS from IT distribution within the same broad Forbes category

This became the Quarto analysis (D1) and presentation (D4), rewritten onto the Forbes/EV-Revenue pipeline once the app itself had moved past the Fortune 500 baseline.

**Phase 4 — Data-correctness fix and feature expansion (renamed "Valuatorix"):** Once the app was in regular use, a suspicion that foreign-currency figures "look wrong" led to discovering a genuine data-quality bug (see "Caught a currency-conversion bug" below) that had silently corrupted both live lookups and the training data itself for ~170 non-USD companies — fixed and the training data corrected in place. This phase also added: a second trained regressor for EV/EBITDA; a 5-factor comparable-company scoring scheme with industry clustering (widening the peer pool to related industries, not just exact matches); a 30%-model/70%-peer-median blended verdict, replacing a straight model-only comparison; an explicit "no market to compare against" framing for private/manually-entered companies (no undervalued/overvalued label where there's no market price); a manual-company snapshot workflow; website-URL description scraping; a full Excel report export; and a display-unit toggle (Billions/Millions/Thousands) applied consistently across the sidebar, all dashboard figures, and the comps table.

---

## How I verified the AI's output

The AI can generate plausible-looking code that fails silently or embeds subtle bugs. I applied the following verification steps:

1. **Syntax check first.** Every Python file was checked with `python -m py_compile` before running. This catches the most obvious errors immediately.

2. **Read before trusting.** For the Quarto report in particular, I reviewed each code cell's logic against the data columns I could see — for example, verifying that the classification target encoding (`global_500 == "yes"`) matched the actual string values in the CSV.

3. **Caught a data-leakage risk.** The AI initially suggested including `profit_margin = profit_mil / revenue_mil` as a feature for the regression task. I flagged this as leakage (the target `profit_mil` is embedded in the feature) and instructed it not to include that feature for the regression pipeline. This is the kind of subtle error AI code generation can introduce.

4. **Caught severe overfitting.** The first version of the Forbes model had CV R² = 0.10 with a train/CV gap of 0.70 — clear overfitting. I diagnosed this and directed the fix: log-transform the EV/Revenue target (reduces right-skew), lower max_depth from 4 to 3, add subsample=0.8 and min_samples_leaf=10. Result: CV R² = 0.45, gap = 0.11.

5. **Caught a financial sector EV error.** Yahoo Finance reports customer deposits as "total debt" for banks, making Enterprise Value negative (e.g. Deutsche Bank). I identified the root cause and directed the fix: for Banking, Insurance, and Diversified Financials, set net_debt = 0 and use Market Cap as EV proxy — consistent with how practitioners treat these sectors.

6. **Verified EV consistency across training and inference.** The Forbes training pool and the live company lookup both need to compute EV the same way for financial companies. I caught that the training data had the fix but the live lookup did not — directed the fix to apply the same FINANCIAL_SECTORS exception in `fetch_ticker()`.

7. **Architecture diagram self-check.** For the interim deliverable, I ran the professor's checklist explicitly and caught two architectural errors (Yahoo Finance routing through the frontend instead of the backend endpoint, and a missing "prediction endpoint" label). Both were fixed before submission.

8. **Caught drift between deliverables after the Forbes pivot.** After rebuilding the live app (D2) on Forbes Global 2000, I asked Claude Code to check whether D1 (`analysis.qmd`) and D4 (`presentation.qmd`) still matched what the app actually trains on. They didn't — both were still analyzing the earlier Fortune 500 / P/S baseline, `presentation.qmd`'s title already claimed the Forbes framing while its code cells didn't, and the architecture diagram still depicted an abandoned Capital IQ / offline-model design. I directed a full rewrite of D1/D4 onto the Forbes/EV-Revenue pipeline, added the feature-selection step the assignment requires (RFE, applied leakage-safely), and regenerated the architecture diagram to match the real pipeline — the kind of cross-document consistency check that's easy to skip when each deliverable is edited independently over time.

9. **Caught a currency-conversion bug corrupting the training data, not just live lookups.** I'd noticed foreign-currency figures "looked wrong" for some companies and asked Claude Code to investigate rather than accepting a guess. It found that Yahoo Finance reports non-USD companies' financial fields in local currency, not USD — the app was reading these raw and unconverted. It verified this concretely against a real company (Toyota's revenue computing at ~$50 trillion instead of ~$314 billion) before proposing a fix, then went further and checked whether the bug had also corrupted the stored *training data* — it had, for ~170 Japanese/Korean companies' `net_debt_B`. Rather than just patching the live code, I had it run a targeted, no-Search-API recompute pass against the existing dataset (avoiding a full re-scrape that would have discarded already-resolved company matches and re-hit a rate limit the project had already fought with before) and back up the data first. I verified the fix myself with a hand calculation (JPY→USD rate × raw revenue ≈ Toyota's real revenue) before accepting it.

10. **Caught a would-be P0 crash before it shipped, via multi-persona document review.** When planning a batch of UI changes, I had Claude Code run a structured review pass (five reviewer personas — coherence, feasibility, scope, design, adversarial) against its own plan before implementing. Two independent reviewers flagged the same real bug: a planned "manual company" feature would have crashed with a `KeyError` on the very first use, because the new code path didn't set a field (`mktcap_B`) that four other places in the app read unconditionally. This was caught and fixed at the planning stage, before any code was written — cheaper than catching it after implementation.

11. **Caught a Streamlit-specific state bug through direct experimentation, not just code reading.** A later plan proposed rekeying sidebar widgets to fix an ordering problem. Two reviewers (independently) flagged a *different* risk in that same fix: Streamlit widgets with a fixed `key=` stop responding to a changed default value on rerun — which would have silently broken the already-working "switch tickers and see the fields update" behavior. Rather than trust the claim or my own reasoning about it, I had Claude Code write a small standalone Streamlit script and literally click through it (switching between two fake companies) to observe the freeze happen, then applied the fix (namespacing widget keys by which company is loaded, not just a fixed name) and re-ran the same experiment to confirm it was resolved. This is the kind of framework-specific behavior that's easy to get wrong by reasoning alone and cheap to verify by just running it.

12. **Verified UI/state changes with Streamlit's own test framework, not just eyeballing a browser.** For several later features, browser-based screenshots proved unreliable (stale page state after navigation). Rather than accept unreliable evidence, I had Claude Code switch to Streamlit's `AppTest` framework — a way to drive the actual app.py programmatically (fill in fields, click buttons, inspect the resulting widget state and page text) without a browser at all. This caught, for example, that a manual company snapshot correctly derived EBITDA from an entered margin percentage, and that switching between two real tickers correctly refreshed every affected sidebar field.

---

## Rough sense of cost and effort

| Phase | Human effort | AI contribution |
|-------|-------------|----------------|
| Dataset choice (Fortune 500) | ~30 min discussion | Research and trade-off analysis |
| Forbes Global 2000 pivot decision | ~20 min discussion | Architecture options and trade-offs |
| Architecture diagram (interim) | ~45 min | Full Mermaid source, rendering command, two bug fixes |
| Plan creation | ~20 min review | Full implementation plan document |
| Fortune 500 analysis (D1, D4) | ~45 min review + edits | Full code generation |
| Forbes G2000 app — core | ~30 min review | Full code generation |
| EV/Revenue migration (P/S → EV/Rev) | ~15 min direction | Full refactor |
| Overfitting diagnosis + fix | ~20 min (identified issue, directed fix) | Executed fix |
| Description enrichment pipeline | ~20 min design | Full script generation |
| Financial sector EV fix | ~10 min (diagnosed bug) | Executed fix |
| Supporting docs | ~20 min | First draft of all |
| Currency-conversion bug: diagnosis + fix + data correction | ~20 min (raised the suspicion, verified the fix against a real company by hand) | Root-cause investigation, fix across both the live and training-data code paths, targeted data recomputation |
| Comps-scoring overhaul (5-factor + industry clustering) + EV/EBITDA model + blended verdict | ~30 min direction + review of the plan's doc-review findings | Planning, multi-persona review, implementation, verification |
| Manual snapshot, URL scraping, Excel export, unit toggle, UI polish (multiple rounds) | ~40 min direction + spot-checks | Planning, implementation, verification (including the Streamlit widget-freeze bug caught and fixed mid-stream) |
| Deliverables realignment (this pass) | ~15 min direction | Cross-checked all 5 deliverables against the assignment rubric and the current app, updated stale figures/methodology, re-rendered D1/D4 |
| **Total human** | **~6 hours** | — |

The AI reduced the raw coding and writing time by roughly 75–80%. The remaining human effort was mostly: making product decisions (which dataset, which features to include or exclude, which metric to use), verifying correctness, and catching edge cases the AI missed (leakage, overfitting, financial sector EV calculation).

The two areas where AI was least helpful:
1. Figuring out that Capital IQ data cannot go in a public GitHub repo — a constraint I had to surface through conversation rather than having the AI anticipate it automatically.
2. Diagnosing model overfitting — the AI generated the model, but recognising that CV R² = 0.10 was unacceptably poor and diagnosing the cause (right-skewed target, overly complex trees) required human judgment about what "good enough" looks like for this task.
