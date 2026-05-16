"""
Drug Overdose MCP Dashboard
Tab 1 — Dataset Recommender (Claude API + web_search MCP)
Tab 2 — K-Means Cluster Predictor (trained model)
Tab 3 — Prophet Forecast (trained model)

Run:
    streamlit run src/mcp_dataset_recommender.py
"""

import streamlit as st
import anthropic
import json
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

# ─────────────────────────────────────────────
# ENV & CONFIG
# ─────────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
api_key = os.getenv("ANTHROPIC_API_KEY")

st.set_page_config(
    page_title="Drug Overdose MCP Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0D1117; }
    .block-container { padding-top: 2rem; }
    [data-testid="stSidebar"] { background-color: #1A1A2E; border-right: 1px solid #2a2a4a; }
    [data-testid="stSidebar"] * { color: #CCCCCC !important; }
    h1 { color: #00D4FF !important; font-size: 1.8rem !important; }
    h2, h3 { color: #FFFFFF !important; }
    p, li, .stMarkdown { color: #CCCCCC !important; }

    [data-testid="stMetric"] {
        background: #1A1A2E; border: 1px solid #2a2a4a;
        border-radius: 10px; padding: 1rem;
    }
    [data-testid="stMetricLabel"] { color: #888 !important; font-size: 0.75rem !important; }
    [data-testid="stMetricValue"] { color: #00D4FF !important; font-size: 1.6rem !important; }

    [data-testid="stExpander"] {
        background: #1A1A2E; border: 1px solid #2a2a4a; border-radius: 10px;
    }
    .streamlit-expanderHeader { color: #FFFFFF !important; font-weight: 600 !important; }

    .stButton > button {
        background: linear-gradient(135deg, #0066CC, #6600CC);
        color: white !important; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-weight: 600; font-size: 1rem;
        transition: all 0.2s; width: 100%;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(0,180,255,0.3); }

    .stTextInput input, .stSelectbox select {
        background: #1A1A2E !important; color: white !important;
        border: 1px solid #2a2a4a !important; border-radius: 8px !important;
    }
    .stNumberInput input {
        background: #1A1A2E !important; color: white !important;
        border: 1px solid #2a2a4a !important;
    }

    .tag { display:inline-block; padding:2px 10px; border-radius:20px;
           font-size:0.72rem; font-weight:700; letter-spacing:0.05em; margin-right:6px; }
    .tag-socio  { background:#1a2744; color:#60a5fa; border:1px solid #3b82f6; }
    .tag-health { background:#1a2e1a; color:#4ade80; border:1px solid #22c55e; }
    .tag-law    { background:#2d1a1a; color:#f87171; border:1px solid #ef4444; }
    .tag-demo   { background:#2a1a2e; color:#c084fc; border:1px solid #a855f7; }
    .tag-policy { background:#2d2200; color:#fbbf24; border:1px solid #f59e0b; }
    .tag-easy   { background:#1a2e1a44; color:#4ade80; border:1px solid #22c55e55; }
    .tag-medium { background:#2d220044; color:#fbbf24; border:1px solid #f59e0b55; }
    .tag-hard   { background:#2d1a1a44; color:#f87171; border:1px solid #ef444455; }

    .ds-card { background:#1A1A2E; border:1px solid #2a2a4a; border-radius:12px;
               padding:1.2rem 1.4rem; margin-bottom:1rem; }
    .ds-name { font-size:1.05rem; font-weight:700; color:#FFFFFF; margin:6px 0 2px; }
    .ds-src  { font-size:0.8rem; color:#666; margin-bottom:10px; }
    .ds-label { font-size:0.75rem; color:#888; font-weight:600;
                text-transform:uppercase; letter-spacing:0.06em; margin:10px 0 4px; }
    .ds-join  { font-family:monospace; font-size:0.82rem; color:#7dd3fc;
                background:#0f172a; border:1px solid #1e293b;
                border-radius:6px; padding:6px 12px; margin-bottom:4px; }
    .ds-enrich { font-size:0.88rem; color:#CCCCCC; line-height:1.6; }
    .ds-url   { font-size:0.8rem; color:#00D4FF; }

    .result-card { background:#1A1A2E; border:1px solid #2a2a4a;
                   border-radius:12px; padding:1.4rem; margin:1rem 0; }
    .cluster-low  { border-color:#FFD700 !important; }
    .cluster-mod  { border-color:#FF8C00 !important; }
    .cluster-high { border-color:#FF4444 !important; }

    .status-box { background:#0f1729; border:1px solid #1e3a5f;
                  border-radius:10px; padding:1rem 1.4rem;
                  margin:1rem 0; color:#7dd3fc; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / "models"
TAGS = Path(__file__).resolve().parent.parent / "reports"


@st.cache_resource
def load_models():
    models = {}
    try:
        # kmeans_model.pkl is a DICT containing kmeans + scaler + metadata
        with open(BASE / 'kmeans_model.pkl', 'rb') as f:
            model_data = pickle.load(f)

        models['kmeans']        = model_data['kmeans']
        models['scaler']        = model_data['scaler']
        models['feature_names'] = model_data['feature_names']
        models['labels']        = {str(k): v for k, v in model_data['label_names'].items()}
        models['kmeans_loaded'] = True

    except Exception as e:
        models['kmeans_loaded'] = False
        models['kmeans_error']  = str(e)

    try:
        with open(BASE / 'prophet_models.pkl', 'rb') as f:
            models['prophet'] = pickle.load(f)
        models['prophet_loaded'] = True
    except Exception as e:
        models['prophet_loaded'] = False
        models['prophet_error']  = str(e)

    return models

models = load_models()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💊 MCP Dashboard")
    st.markdown("---")
    st.markdown("### 🤖 Model Status")
    if models.get('kmeans_loaded'):
        st.success("✓ K-Means loaded")
    else:
        st.error(f"✗ K-Means: {models.get('kmeans_error','not found')}")
    if models.get('prophet_loaded'):
        st.success(f"✓ Prophet loaded ({len(models['prophet'])} states)")
    else:
        st.error(f"✗ Prophet: {models.get('prophet_error','not found')}")
    if api_key:
        st.success("✓ API Key loaded")
    else:
        st.error("✗ API Key missing")
    st.markdown("---")
    st.markdown("### 📂 Dataset Info")
    st.markdown("""
    - **Records:** 47,365
    - **Period:** 2015 – 2025
    - **States:** 50
    - **Indicators:** 10 drug types
    - **Clusters:** 3 (K-Means)
    """)
    st.markdown("---")
    st.caption("Phase 2 · MCP Deployment Demo")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a data engineering and public health research expert.
The user has a US drug overdose deaths dataset:
- Columns: State, State Name, Indicator (drug type), Date (monthly 2015-2025), Death Count (12-month rolling sum)
- 10 drug indicators: Cocaine, Heroin, Methadone, Natural opioids, Synthetic opioids, Psychostimulants + combined
- Coverage: 50 US states/territories
- K-Means clustering done (K=3): Low Volume/Rural, Moderate & Rising, High Burden Crisis
Search the web and recommend exactly 5 real, publicly available datasets that can be joined with this dataset.
For each dataset return a JSON object with fields: name, source, url, joinKey, enrichment, category (one of Socioeconomic/Healthcare/Law Enforcement/Demographics/Policy), difficulty (Easy/Medium/Hard).
Return ONLY a valid JSON array of 5 objects, no markdown, no explanation."""

CATEGORY_ICONS   = {"Socioeconomic":"💰","Healthcare":"🏥","Law Enforcement":"⚖️","Demographics":"👥","Policy":"📋"}
CATEGORY_TAG     = {"Socioeconomic":"tag-socio","Healthcare":"tag-health","Law Enforcement":"tag-law","Demographics":"tag-demo","Policy":"tag-policy"}
DIFFICULTY_TAG   = {"Easy":"tag-easy","Medium":"tag-medium","Hard":"tag-hard"}
DIFFICULTY_EMOJI = {"Easy":"🟢","Medium":"🟡","Hard":"🔴"}
CLUSTER_COLORS   = {
    "Low Volume / Rural": ("#FFD700", "cluster-low",  "🟡"),
    "Moderate & Rising":  ("#FF8C00", "cluster-mod",  "🟠"),
    "High Burden Crisis": ("#FF4444", "cluster-high", "🔴"),
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def render_dataset_card(ds, idx):
    cat      = ds.get("category", "Socioeconomic")
    diff     = ds.get("difficulty", "Medium")
    cat_cls  = CATEGORY_TAG.get(cat, "tag-socio")
    diff_cls = DIFFICULTY_TAG.get(diff, "tag-medium")
    icon     = CATEGORY_ICONS.get(cat, "📊")
    d_emoji  = DIFFICULTY_EMOJI.get(diff, "🟡")
    st.markdown(f"""
    <div class="ds-card">
        <span class="tag {cat_cls}">{icon} {cat}</span>
        <span class="tag {diff_cls}">{d_emoji} {diff}</span>
        <div class="ds-name">{idx}. {ds.get('name','')}</div>
        <div class="ds-src">📦 {ds.get('source','')}</div>
        <div class="ds-label">🔗 Join Key</div>
        <div class="ds-join">{ds.get('joinKey','')}</div>
        <div class="ds-label">📊 What This Enables</div>
        <div class="ds-enrich">{ds.get('enrichment','')}</div>
        <div style="margin-top:10px">
            <a href="{ds.get('url','#')}" target="_blank" class="ds-url">🌐 {ds.get('url','')}</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


def call_claude_with_web_search(key):
    client   = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content":
            "Search the web and find 5 real publicly available datasets "
            "I can join with my US drug overdose deaths dataset. Return only JSON array."}]
    )
    text_block = next((b for b in response.content if b.type == "text"), None)
    if not text_block:
        raise ValueError("No text response from Claude API")
    raw   = text_block.text.strip().replace("```json","").replace("```","").strip()
    start = raw.find("[")
    end   = raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("Could not parse JSON from response")
    return json.loads(raw[start:end+1])


def predict_cluster(cocaine, heroin, methadone, natural_opioids,
                    synthetic_opioids, psychostimulants, trend_slope):
    features = np.array([[
        cocaine, heroin, methadone,
        natural_opioids, synthetic_opioids,
        psychostimulants, trend_slope
    ]])
    # Use scaler (extracted from kmeans_model.pkl dict)
    features_scaled = models['scaler'].transform(features)
    cluster_id      = models['kmeans'].predict(features_scaled)[0]
    cluster_label   = models['labels'][str(cluster_id)]
    return int(cluster_id), cluster_label


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 💊 Drug Overdose MCP Dashboard")
st.markdown("K-Means clustering · Prophet forecasting · MCP web search — all in one place.")
st.markdown("""
<div style="display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 20px">
  <span class="tag tag-socio">📅 2015–2025</span>
  <span class="tag tag-health">🗺️ 50 States</span>
  <span class="tag tag-demo">💊 10 Indicators</span>
  <span class="tag tag-policy">🔬 K=3 Clusters</span>
  <span class="tag tag-law">🔗 State + Month Join</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Dataset Recommender (MCP)",
    "🎯 K-Means Cluster Predictor",
    "📈 Prophet Forecast",
])

# ═══════════════════════════════
# TAB 1 — DATASET RECOMMENDER
# ═══════════════════════════════
with tab1:
    st.markdown("### 🔍 Find Linkable Datasets via MCP Web Search")
    st.markdown("Uses **Claude API + web_search MCP tool** to find real datasets joinable with your overdose data.")
    st.divider()

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        run_btn = st.button("🚀 Find Linkable Datasets via MCP", key="mcp_btn")

    if "datasets"  not in st.session_state: st.session_state.datasets  = []
    if "mcp_error" not in st.session_state: st.session_state.mcp_error = None

    if run_btn:
        if not api_key:
            st.error("⚠️ API key not found in .env file.")
        else:
            st.session_state.datasets  = []
            st.session_state.mcp_error = None
            steps = [
                "🔌 Connecting to Claude API...",
                "🌐 Invoking web_search MCP tool...",
                "🔎 Searching CDC repositories...",
                "🔎 Searching Census & socioeconomic datasets...",
                "🔎 Searching healthcare & policy databases...",
                "⚙️ Evaluating join compatibility...",
                "📊 Ranking by analytical value...",
            ]
            status_box = st.empty()
            progress   = st.progress(0)
            try:
                for i, step in enumerate(steps[:-1]):
                    status_box.markdown(f'<div class="status-box">⏳ {step}</div>', unsafe_allow_html=True)
                    progress.progress((i+1)/len(steps))
                    time.sleep(0.4)
                results = call_claude_with_web_search(api_key)
                progress.progress(1.0)
                status_box.markdown('<div class="status-box" style="border-color:#22c55e;color:#4ade80">✅ Done!</div>', unsafe_allow_html=True)
                time.sleep(0.6)
                status_box.empty()
                progress.empty()
                st.session_state.datasets = results
            except Exception as e:
                progress.empty()
                status_box.empty()
                st.session_state.mcp_error = str(e)

    if st.session_state.mcp_error:
        st.error(f"❌ {st.session_state.mcp_error}")

    if st.session_state.datasets:
        datasets = st.session_state.datasets
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Found",  len(datasets))
        m2.metric("Easy",   sum(1 for d in datasets if d.get("difficulty")=="Easy"))
        m3.metric("Medium", sum(1 for d in datasets if d.get("difficulty")=="Medium"))
        m4.metric("Hard",   sum(1 for d in datasets if d.get("difficulty")=="Hard"))
        st.divider()
        for i, ds in enumerate(datasets, 1):
            render_dataset_card(ds, i)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("🔄 Search Again", key="re_search"):
                st.session_state.datasets = []
                st.rerun()

# ═══════════════════════════════
# TAB 2 — K-MEANS PREDICTOR
# ═══════════════════════════════
with tab2:
    st.markdown("### 🎯 K-Means Cluster Predictor")
    st.markdown("Enter average monthly deaths per drug type to predict which cluster a state belongs to.")
    st.divider()

    if not models.get('kmeans_loaded'):
        st.error(f"❌ K-Means model not loaded: {models.get('kmeans_error')}")
        st.info("Make sure `kmeans_model.pkl`, `pc.pkl`, and `label_names.json` are in the same folder as this script.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📋 State Info")
            state_name  = st.text_input("State Name", value="Ohio", key="km_state")
            trend_slope = st.number_input(
                "Trend Slope (deaths/month change)",
                value=0.5, min_value=-5.0, max_value=10.0, step=0.1,
                help="Positive = rising, Negative = declining", key="km_trend"
            )
        with col2:
            st.markdown("#### 💊 Avg Monthly Deaths by Drug Type")
            cocaine           = st.number_input("Cocaine",                          value=8.0,  min_value=0.0, step=0.5, key="km_coc")
            heroin            = st.number_input("Heroin",                           value=12.0, min_value=0.0, step=0.5, key="km_her")
            methadone         = st.number_input("Methadone",                        value=3.0,  min_value=0.0, step=0.5, key="km_meth")
            natural_opioids   = st.number_input("Natural & Semi-synthetic Opioids", value=15.0, min_value=0.0, step=0.5, key="km_nat")
            synthetic_opioids = st.number_input("Synthetic Opioids (excl. Methadone)", value=45.0, min_value=0.0, step=0.5, key="km_syn")
            psychostimulants  = st.number_input("Psychostimulants",                 value=6.0,  min_value=0.0, step=0.5, key="km_psy")

        st.divider()
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            predict_btn = st.button("🎯 Predict Cluster", key="km_predict")

        if predict_btn:
            try:
                cluster_id, cluster_label = predict_cluster(
                    cocaine, heroin, methadone,
                    natural_opioids, synthetic_opioids,
                    psychostimulants, trend_slope
                )
                total = cocaine + heroin + methadone + natural_opioids + synthetic_opioids + psychostimulants
                color, css_class, emoji = CLUSTER_COLORS.get(cluster_label, ("#00D4FF","","🔵"))

                st.markdown(f"""
                <div class="result-card {css_class}">
                    <div style="font-size:0.8rem;color:#888;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">PREDICTION RESULT</div>
                    <div style="font-size:2rem;font-weight:800;color:{color};margin-bottom:4px">{emoji} {cluster_label}</div>
                    <div style="font-size:0.9rem;color:#CCCCCC;margin-bottom:16px">Cluster ID: {cluster_id}</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:12px">
                        <div style="background:#0f172a;border-radius:8px;padding:10px 14px">
                            <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase">Total Deaths/Mo</div>
                            <div style="font-size:1.4rem;font-weight:700;color:{color}">{total:.1f}</div>
                        </div>
                        <div style="background:#0f172a;border-radius:8px;padding:10px 14px">
                            <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase">Trend Slope</div>
                            <div style="font-size:1.4rem;font-weight:700;color:{'#f87171' if trend_slope>0 else '#4ade80'}">{trend_slope:+.2f}</div>
                        </div>
                        <div style="background:#0f172a;border-radius:8px;padding:10px 14px">
                            <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase">Trajectory</div>
                            <div style="font-size:1.4rem;font-weight:700;color:{'#f87171' if trend_slope>0 else '#4ade80'}">{'↑ Rising' if trend_slope>0 else '↓ Declining'}</div>
                        </div>
                    </div>
                    <div style="margin-top:14px;font-size:0.88rem;color:#94a3b8;line-height:1.6;background:#0f172a;border-radius:8px;padding:12px 14px">
                        <b style="color:#CCCCCC">{state_name}</b> is classified as 
                        <b style="color:{color}">{cluster_label}</b> with {total:.1f} avg monthly deaths. 
                        Trend of {trend_slope:+.2f} deaths/month indicates a 
                        {'rising' if trend_slope>0 else 'declining'} trajectory.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 📊 Drug Type Breakdown")
                chart_data = pd.DataFrame({
                    "Drug Type": ["Cocaine","Heroin","Methadone","Natural Opioids","Synthetic Opioids","Psychostimulants"],
                    "Avg Deaths/Month": [cocaine, heroin, methadone, natural_opioids, synthetic_opioids, psychostimulants]
                }).sort_values("Avg Deaths/Month", ascending=True)
                st.bar_chart(chart_data.set_index("Drug Type"))

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

# ═══════════════════════════════
# TAB 3 — PROPHET FORECAST
# ═══════════════════════════════
with tab3:
    st.markdown("### 📈 Prophet Forecast")
    st.markdown("Select a state and forecast period to see projected monthly overdose deaths.")
    st.divider()

    if not models.get('prophet_loaded'):
        st.error(f"❌ Prophet models not loaded: {models.get('prophet_error')}")
        st.info("Make sure `prophet_models.pkl` is in the same folder as this script.")
    else:
        available_states = sorted(models['prophet'].keys())
        col1, col2 = st.columns([2,1])
        with col1:
            selected_state = st.selectbox(
                "Select State", options=available_states,
                index=available_states.index("Ohio") if "Ohio" in available_states else 0,
                key="prophet_state"
            )
        with col2:
            periods = st.slider("Forecast Months", min_value=6, max_value=36, value=12, step=6, key="prophet_periods")

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            forecast_btn = st.button("📈 Generate Forecast", key="prophet_btn")

        if forecast_btn:
            with st.spinner(f"Running Prophet forecast for {selected_state}..."):
                try:
                    prophet_model = models['prophet'][selected_state]
                    
                    historical = prophet_model.history.copy()
                    hist_trend = "Rising" if historical['y'].iloc[-1] < historical['y'].iloc[0] else "Declining"
                    
                    future        = prophet_model.make_future_dataframe(periods=periods, freq='MS')
                    forecast      = prophet_model.predict(future)

                    forecast_only = forecast.tail(periods)[['ds','yhat','yhat_lower','yhat_upper']].copy()
                    peak_val      = forecast_only['yhat'].max()
                    peak_month    = forecast_only.loc[forecast_only['yhat'].idxmax(), 'ds'].strftime('%b %Y')
                    avg_val       = forecast_only['yhat'].mean()
                    fut_trend_dir  = "Rising" if forecast_only['yhat'].iloc[-1] > forecast_only['yhat'].iloc[0] else "Declining"
                    
                    trend_warning = None
                    if hist_trend != fut_trend_dir:
                        trend_warning = f"⚠️ Note: Historical trend is {hist_trend.lower()}, but forecast shows {fut_trend_dir.lower()}"

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Peak Forecast",  f"{peak_val:.0f}/mo")
                    m2.metric("Peak Month",      peak_month)
                    m3.metric("Avg Forecast",   f"{avg_val:.0f}/mo")
                    m4.metric("Trend Direction", "📈 Rising" if fut_trend_dir == "Rising" else "📉 Declining")

                    if trend_warning:
                        st.warning(trend_warning)

                    st.divider()
                    st.markdown(f"#### 📊 {selected_state} — {periods}-Month Forecast")

                    fore_chart = forecast_only.rename(columns={
                        'ds':'Date','yhat':'Forecast',
                        'yhat_lower':'Lower Bound','yhat_upper':'Upper Bound'
                    }).set_index('Date')
                    st.line_chart(fore_chart[['Forecast','Lower Bound','Upper Bound']])

                    st.divider()
                    st.markdown("#### 📋 Forecast Table")
                    display_df = forecast_only.copy()
                    display_df['ds']         = display_df['ds'].dt.strftime('%b %Y')
                    display_df['yhat']       = display_df['yhat'].round(1)
                    display_df['yhat_lower'] = display_df['yhat_lower'].round(1)
                    display_df['yhat_upper'] = display_df['yhat_upper'].round(1)
                    display_df.columns       = ['Month','Forecast','Lower Bound','Upper Bound']
                    st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

                    st.markdown(f"""
                    <div class="result-card" style="margin-top:1rem">
                        <div style="font-size:0.8rem;color:#888;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px">INTERPRETATION</div>
                        <div style="font-size:0.92rem;color:#CCCCCC;line-height:1.7">
                            Prophet forecasts <b style="color:#00D4FF">{selected_state}</b> will peak at 
                            <b style="color:#FF6B6B">{peak_val:.0f} deaths/month</b> around <b>{peak_month}</b> 
                            with a {periods}-month average of <b>{avg_val:.0f} deaths/month</b>. 
                            Overall forecast trajectory: <b>{"📈 Rising" if fut_trend_dir == "Rising" else "📉 Declining"}</b>.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Forecast failed: {str(e)}")
