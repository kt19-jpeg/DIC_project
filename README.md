# US Drug Overdose Deaths — Analysis & Forecasting (2015–2025)

Analyzing CDC overdose death data across three phases: exploratory analysis → clustering + forecasting → scalable Databricks pipeline.

**Focus areas:** Synthetic opioids (fentanyl), psychostimulants (meth), cocaine, state-level trends

---

## The Goal

Build a state-level early warning system that forecasts overdose deaths 1–3 months ahead — by drug category and state — to support resource allocation and public health planning.

---

## Data

**CDC VSRR Provisional Drug Overdose Death Counts** — monthly, 12-month rolling totals, 50k+ rows, state + national level.

> ⚠️ The `Death Count` column is a **12-month rolling sum**, not actual monthly deaths. All scripts unroll it first:
> `actual_monthly(t) = rolling(t) - rolling(t-12)`

---

## Phases

### Phase 1 — Exploration & Cleaning
```bash
python data_findings.py   # adjust relative path
python data_clean.py      # or use the notebook version
```

---

### Phase 2 — Clustering + Forecasting + MCP Dashboard

**Setup**
```bash
git clone <repo-url> && cd DIC_project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

**Run in order** (or skip — models are pre-trained in `reports/`)
```bash
python src/clustering.py          # K-Means (K=3), saves models + charts to reports/
python src/prophet_training.py    # Trains ~49 Prophet models, takes a few minutes
```

**Launch dashboard**
```bash
streamlit run src/mcp_dataset_recommender.py
# opens at http://localhost:8501
```

| Tab | What it does |
|-----|-------------|
| Dataset Recommender | Live web search via Claude API to find linkable datasets |
| K-Means Predictor | Enter drug death values → get cluster prediction |
| Prophet Forecast | Select state + horizon → monthly death projections |

**Cluster labels**

| Cluster | Label | Avg Deaths/Month |
|---------|-------|-----------------|
| 0 | Low Volume / Rural | ~81 |
| 2 | Moderate & Rising | ~200 |
| 1 | High Burden / Crisis | ~362 |

---

### Phase 3 — Databricks + Medallion Architecture

Re-implements Phase 2 at scale using Spark on Databricks with Delta Lake throughout.

**Additional data:** KFF State Health Facts — OUD prevalence by state (SAMHSA NSDUH 2022–2023)
```bash
python src/convert_numbers_to_csv.py  # converts .numbers → CSV before DBFS upload
```

**Medallion layers**

| Layer | Tables | What happens |
|-------|--------|-------------|
| Bronze | `bronze_cdc_overdose`, `bronze_kff_opioid_disorder` | Raw ingestion, no transforms |
| Silver | `silver_*` | Type casting, quality filters (`pct_complete == 100`, `pct_pending < 0.3`), dedup |
| Gold | 7 tables incl. `gold_ml_features`, `gold_cdc_kff_joined` | Aggregates, YoY metrics, CDC×KFF join, ML-ready features |

**MLlib models** (all use Pipeline API + CrossValidator)

| Model | Task |
|-------|------|
| K-Means | State overdose burden clustering |
| Ridge Regression | Annual death count prediction (uses OUD prevalence features) |
| Random Forest | High vs low burden classification |

**Run notebooks in this order:**
```
p3_bronze_layer.ipynb → p3_silver_layer.ipynb → p3_gold_layer.ipynb → p3_mllib_models.ipynb
```
Database `eas587_phase3` is created automatically in the first notebook.

---

## Common Issues

**`kmeans_model.pkl` not found** — run `clustering.py` first

**`401 authentication_error`** — check `.env`: no quotes around the key, no trailing spaces

**`dict object has no attribute predict`** — use the updated `load_models()` that extracts `model_data['kmeans']` and `model_data['scaler']` separately

**Prophet install fails** — install `pystan` first, then `prophet`

---

## Dependencies

```
pyspark>=3.3.0
delta-spark>=2.0.0
numbers-parser   # local only, for KFF .numbers conversion
```

---

## References

- Seth P, et al. *Clarifying CDC's Efforts to Quantify Overdose Deaths.* Public Health Rep. 2023. [doi:10.1177/00333549221123586](https://doi.org/10.1177/00333549221123586)
- [https://journals.sagepub.com/doi/10.1177/19427891251401921](https://journals.sagepub.com/doi/10.1177/19427891251401921)
