# US Overdose Death Trends (2015–2025): Forecasting & Early Warning System

## Overview
This project analyzes US drug overdose death trends from **2015–2025**, with a focus on:
- **Synthetic opioids (fentanyl)**
- **Psychostimulants (methamphetamine)**
- **Cocaine**
- **State-level shifts over time**

The long-term end goal is to build a:

> **State-level overdose forecasting model + early warning system**

This system aims to predict overdose deaths **1–3 months ahead** and identify which drug categories are driving future increases.

---

## Motivation & Problem Context
Drug overdose deaths remain a major public health crisis in the United States. In recent years, trends have shifted dramatically due to:
- the rapid spread of fentanyl
- rising stimulant-related deaths
- drug combinations increasing lethality

Understanding these patterns and forecasting near-future risk can support:
- resource allocation
- public health interventions
- prevention and harm-reduction planning

---

## Data Source
This project uses the CDC dataset:

**VSRR Provisional Drug Overdose Death Counts (Monthly, 12-month rolling totals)**

The dataset includes:
- 50,000+ rows (meets project requirements)
- state-level and national-level breakdowns
- multiple drug categories
- completeness and reporting-delay metadata

---

## End Goal (Final Phase)
###  State-Level Overdose Forecasting + Early Warning System
The final system will:
1. Forecast overdose deaths by state (1–3 months ahead)
2. Predict category-specific outcomes:
   - synthetic opioids (fentanyl)
   - psychostimulants
   - cocaine
   - total overdose deaths
3. Produce an interpretable risk ranking of states
4. Explain drivers of predicted risk (feature importance / explainability)

---

## Quick Setup for reproducibility

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 2. Install dependencies (after you create requirements.txt)
pip install -r requirements.txt

# Files to run
run data_findings.py with changing the relative path accordingly

run data_clean.py for cleaning the data , have also attached notebook for reference which does the same cleaning process as .py file

```
## Refrences

Seth P, Baldwin GT, Davis NL, Jones CM. Clarifying CDC's Efforts to Quantify Overdose Deaths. Public Health Rep. 2023 Sep-Oct;138(5):721-726. doi: 10.1177/00333549221123586. Epub 2022 Oct 1. PMID: 36184930; PMCID: PMC10467501.


https://journals.sagepub.com/doi/10.1177/19427891251401921


# Drug Overdose Deaths — Phase 2
**CDC Drug Overdose Deaths Analysis | K-Means Clustering + Prophet Forecasting + MCP Deployment**


## Prerequisites

- Python 3.9+
- Virtual environment (recommended)
- Anthropic API key — get one at [console.anthropic.com](https://console.anthropic.com)

---

## Setup

**1. Clone the repo and create a virtual environment**
```bash
git clone <your-repo-url>
cd DIC_project
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your API key**

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=sk-ant-api03-...
```


---

## Reproducing the Results

Run these scripts **in order**. Each step generates files that the next step depends on.

# NOTE:Skip this part as we already have the models in the models directory  

### Step 1 — Run Clustering

Unrolls CDC rolling sums, engineers features, fits K-Means (K=3), and saves all model artifacts to `reports/`.

```bash
python src/clustering.py
```

**Outputs:**
- `reports/kmeans_model.pkl` — fitted K-Means + scaler + metadata
- `reports/pca_model.pkl` — PCA transformer
- `reports/label_names.json` — cluster ID to label mapping
- `reports/cluster_assignments.csv` — state-to-cluster mapping
- `reports/01_elbow_silhouette.png`
- `reports/02_clustering_pca.png`
- `reports/03_cluster_profiles.png`
- `reports/04_us_clustering_map.html`
- `reports/05_year_analysis_map.html`
- `reports/06_year_trend_by_cluster.png`

### Step 2 — Train Prophet Models

Fits one Prophet model per state on actual monthly deaths (unrolled). Saves all models as a single pickle file.

```bash
python src/prophet_training.py
```

**Output:**
- `reports/prophet_models.pkl` — dict of `{state_name: Prophet model}`

> This step takes a few minutes — it fits ~49 models.

---

## Running the Dashboard for easy reproducibility

Runing in git Codespaces 

```bash
pip install -r requirements.txt

python -m streamlit run src/mcp_dataset_recommender.py
```




Once both model files exist in `reports/`, launch the Streamlit app:

```bash
streamlit run src/mcp_dataset_recommender.py
```

Opens at `http://localhost:8501` with three tabs:

| Tab | What it does |
|-----|-------------|
| 🔍 Dataset Recommender | Calls Claude API + web_search MCP tool live to find linkable datasets |
| 🎯 K-Means Predictor | Enter drug death values → predicts cluster for any state |
| 📈 Prophet Forecast | Select a state + forecast horizon → projected monthly deaths |

---

## Key Data Note

The CDC `Death Count` column is a **12-month rolling sum**, not actual monthly deaths. All scripts unroll it first:

```python
actual_monthly(t) = rolling(t) - rolling(t - 12 months)
```

Skipping this step produces inflated features and incorrect cluster assignments.

---

## MCP Deployment

The dashboard uses the Anthropic `web_search_20250305` MCP tool in Tab 1 to search the internet live and return structured dataset recommendations. This requires a valid Anthropic API key in your `.env` file.

Tabs 2 and 3 run entirely locally using the trained pickle files — no API key needed for those.

---

## Cluster Labels

| Cluster ID | Label | Avg Deaths/Month |
|-----------|-------|-----------------|
| 0 | Low Volume / Rural | ~81 |
| 2 | Moderate & Rising | ~200 |
| 1 | High Burden Crisis | ~362 |

> Cluster IDs are assigned by K-Means randomly — always verify against `cluster_assignments.csv`.

---

## Common Issues

**`No such file or directory: kmeans_model.pkl`**
Run `clustering.py` first to generate model files.

**`401 authentication_error`**
Check your `.env` file — no quotes around the key, no trailing spaces.

**`Prediction failed: dict object has no attribute predict`**
The app loads `kmeans_model.pkl` as a dict. Make sure you're using the updated `load_models()` function that extracts `model_data['kmeans']` and `model_data['scaler']` separately.

**Prophet install fails**
```bash
pip install pystan
pip install prophet
```
Install `pystan` first, then `prophet`.


---

## Phase 3 — Scalable Pipeline with Databricks & Medallion Architecture

Phase 3 re-implements the Phase 2 analytics pipeline at scale using Apache Spark on Databricks. The entire dataset moves through a **Medallion Architecture** (Bronze → Silver → Gold) backed by Delta Lake, with MLlib replacing the pandas/sklearn stack for distributed model training.

### Additional Data Source

We integrated the **KFF State Health Facts** dataset on Opioid Use Disorder (OUD) prevalence (SAMHSA NSDUH 2022–2023), providing state-level OUD rates split by adolescents (ages 12–17) and adults (18+) for all 50 states and DC. The original file is in Apple Numbers format — run `src/convert_numbers_to_csv.py` to generate the CSV before uploading to DBFS.

### Medallion Architecture

| Layer | Tables | Description |
|---|---|---|
| **Bronze** | `bronze_cdc_overdose`, `bronze_kff_opioid_disorder` | Raw ingestion — no transforms, all columns preserved as strings |
| **Silver** | `silver_cdc_overdose`, `silver_kff_opioid_disorder` | Type casting, quality filter (`pct_complete == 100`, `pct_pending < 0.3`), null drops, deduplication, derived columns |
| **Gold** | `gold_state_annual_metrics`, `gold_indicator_trends`, `gold_ml_features`, `gold_cdc_kff_joined`, 3 insight tables | Business aggregates, YoY window metrics, CDC × KFF join on `state_name`, ML-ready feature table with `high_burden` label |

### MLlib Models

All three models use the **Pipeline API** with **CrossValidator** for hyperparameter tuning.

| Model | Task | Key Features |
|---|---|---|
| K-Means Clustering | Group states by overdose burden profile | `avg_monthly_deaths`, `max_monthly_deaths`, `stddev_deaths`, `yoy_pct_change` |
| Ridge Regression | Predict annual death count per state | CDC features + KFF `adult_oud_pct`, `adolescent_oud_pct`, `combined_oud_pct` |
| Random Forest Classifier | Predict high vs low burden state-year | CDC features + `drug_category` (OHE encoded) |

### Notebooks — Run in this order

```
notebooks/databricks/
├── p3_bronze_layer.ipynb        # Ingest both CSVs → Delta tables
├── p3_silver_layer.ipynb        # Clean, cast, filter, deduplicate
├── p3_gold_layer.ipynb          # Aggregations, ML features, KFF join, insights
└── p3_mllib_models.ipynb        # K-Means, Ridge Regression, Random Forest
```

### Data Setup


2. Upload `cleaned_drug_overdose_deaths.csv` and `kff_opioid_disorder_by_state.csv` to volumes in dBfs
3. Import all `.ipynb` files into Databricks via **Workspace → Import → IPython Notebook**
4. Run notebooks in the order listed above — database `eas587_phase3` is created automatically in notebook 01

> Large data files are excluded from this repository. See `data/README.md` for download instructions or use the UB Box link provided in the submission.

### Dependencies

```
pyspark>=3.3.0
delta-spark>=2.0.0
numbers-parser          # for KFF .numbers → CSV conversion (local only)
```

---

## Authors
Kavyansh
