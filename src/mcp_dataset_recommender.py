"""
Drug Overdose Deaths — K-Means Clustering Analysis
The raw 'Death Count' is a 12-month ROLLING SUM. We unroll it:
    actual_monthly(t) = rolling(t) - rolling(t - 12 months)

FIX: Large states (CA, TX, FL, OH, PA, NY etc.) have NaN for the
"Number of Drug Overdose Deaths" total indicator. The fix is to use
the SUM of 6 base drug-specific indicators per state per month,
which gives coverage for all 50 states.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import pickle


os.makedirs(OUTPUT_DIR, exist_ok=True)

# ───────── STYLING ─────────
COLORS = {'gold': '#FFD700', 'red': '#FF6B6B', 'cyan': '#00D4FF'}
THEME = {'dark': '#0D1117', 'card': '#1A1A2E', 'text': '#CCCCCC', 'accent': '#AAAAAA'}

# ───────── LOAD & PREPROCESS ─────────
processed_data_path = '/Users/kavyansh/DIC_project/data/processed/cleaned_drug_overdose_deaths.csv'
df = pd.read_csv(processed_data_path)
df['Death Count'] = pd.to_numeric(df['Death Count'], errors='coerce')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['State Name', 'Indicator', 'Date'])

# Unroll 12-month rolling sum → actual monthly deaths
df['Actual Monthly Deaths'] = (
    df.groupby(['State Name', 'Indicator'])['Death Count']
    .transform(lambda s: s - s.shift(12))
)
df_actual = df.dropna(subset=['Actual Monthly Deaths'])
df_actual = df_actual[df_actual['Actual Monthly Deaths'] >= 0].copy()


print(f"Dataset: {df.shape[0]} records | {df['State Name'].nunique()} states | "
      f"{df['Date'].min().date()} to {df['Date'].max().date()}")

# ───────── BASE INDICATORS (no double-counting) ─────────
# These 6 are mutually exclusive drug categories.
# Do NOT use combined/rollup indicators like "Opioids (T40.0-T40.4,T40.6)"
# as they overlap with the individual ones and inflate counts.
BASE_INDICATORS = [
    'Cocaine (T40.5)',
    'Heroin (T40.1)',
    'Methadone (T40.3)',
    'Natural & semi-synthetic opioids (T40.2)',
    'Synthetic opioids, excl. methadone (T40.4)',
    'Psychostimulants with abuse potential (T43.6)',
]
df_base = df_actual[df_actual['Indicator'].isin(BASE_INDICATORS)].copy()

# ── State-level monthly total (sum across all 6 drug types) ──
state_monthly = (
    df_base.groupby(['State Name', 'Date'])['Actual Monthly Deaths']
    .sum().reset_index()
)

# ── Per-state average monthly deaths (covers ALL states with any data) ──
state_avg_all = (
    state_monthly.groupby('State Name')['Actual Monthly Deaths']
    .mean().reset_index()
)
state_avg_all.columns = ['State Name', 'Avg Monthly Deaths (All States)']

# ── Trend slope per state ──
def compute_trend(group):
    group = group.sort_values('Date')
    if len(group) < 6:
        return np.nan
    slope, _ = np.polyfit(np.arange(len(group)), group['Actual Monthly Deaths'].values, 1)
    return slope

state_trend_all = (
    state_monthly.groupby('State Name')
    .apply(compute_trend)
    .reset_index()
)
state_trend_all.columns = ['State Name', 'Trend Slope (All States)']

# ───────── CLUSTERING FEATURES ─────────
# Use the same base indicators as a pivot for clustering features
pivot = (
    df_base.groupby(['State Name', 'Indicator'])['Actual Monthly Deaths']
    .mean().unstack(fill_value=0).reset_index()
)
features = pivot.merge(state_trend_all, on='State Name', how='left').dropna()
features = features.rename(columns={'Trend Slope (All States)': 'Trend Slope'})

state_names = features['State Name'].values
X = features.drop('State Name', axis=1).values
X_scaled = StandardScaler().fit_transform(X)
print(f"Features: {features.shape[0]} states × {features.shape[1]-1} indicators\n")

# ───────── FIND OPTIMAL K ─────────
K_range = range(2, 9)
inertias, silhouettes = [], []
print("K-selection scores:")
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouettes.append(sil)
    print(f"  K={k} | Inertia: {km.inertia_:.1f} | Silhouette: {sil:.3f}")

best_k = 3
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = km_final.fit_predict(X_scaled)

# ───────── CLUSTER CHARACTERISTICS ─────────
# Use all-state avg deaths (from base indicators sum) for cluster profiling
features_with_avg = features.merge(state_avg_all, on='State Name', how='left')

cluster_chars = {}
for i in range(best_k):
    mask = cluster_labels == i
    states_in = state_names[mask]
    avg_deaths = features_with_avg[
        features_with_avg['State Name'].isin(states_in)
    ]['Avg Monthly Deaths (All States)'].mean()
    avg_slope = features_with_avg[
        features_with_avg['State Name'].isin(states_in)
    ]['Trend Slope'].mean()
    cluster_chars[i] = {'states': list(states_in), 'deaths': avg_deaths, 'slope': avg_slope}
    print(f"Cluster {i}: {len(states_in):2d} states | avg={avg_deaths:6.1f}/mo | trend={avg_slope:+.3f}")

sorted_by_deaths = sorted(cluster_chars.keys(), key=lambda x: cluster_chars[x]['deaths'])
label_names = {
    sorted_by_deaths[0]: 'Low Volume / Rural',
    sorted_by_deaths[1]: 'Moderate & Rising',
    sorted_by_deaths[2]: 'High Burden Crisis',
}

# ── result_df: per-state actual avg deaths (NOT cluster-level averages) ──
state_actual_avg = features_with_avg[['State Name', 'Avg Monthly Deaths (All States)', 'Trend Slope']].copy()
state_actual_avg.columns = ['State Name', 'Avg Actual Deaths/Mo', 'Trend Slope']

result_df = pd.DataFrame({
    'State':         state_names,
    'Cluster':       cluster_labels,
    'Cluster Label': [label_names[c] for c in cluster_labels],
}).merge(state_actual_avg, left_on='State', right_on='State Name', how='left').drop('State Name', axis=1)

result_df['Avg Actual Deaths/Mo'] = result_df['Avg Actual Deaths/Mo'].round(1)
result_df['Trend (deaths/mo)'] = result_df['Trend Slope'].round(3)
result_df = result_df.drop('Trend Slope', axis=1)
result_df.sort_values('Cluster').reset_index(drop=True).to_csv(
    f'{OUTPUT_DIR}/cluster_assignments.csv', index=False
)
print(f"\n✓ cluster_assignments.csv")
# ───────── SAVE KMEANS MODEL ─────────
model_data = {
    'kmeans': km_final,
    'scaler': StandardScaler().fit(X),
    'feature_names': features.columns[1:].tolist(),
    'state_names': state_names.tolist(),
    'cluster_labels': cluster_labels.tolist(),
    'label_names': label_names,
    'best_k': best_k,
}

with open(f'{OUTPUT_DIR}/kmeans_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print(f"✓ kmeans_model.pkl")
# ───────── PCA ─────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Save PCA as well for future use
with open(f'{OUTPUT_DIR}/pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)
print(f"✓ pca_model.pkl")

# ───────── PLOT 1: ELBOW + SILHOUETTE ─────────
def styled_ax(ax):
    ax.set_facecolor(THEME['dark'])
    ax.tick_params(colors=THEME['text'])
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=THEME['dark'])
for ax in axes:
    styled_ax(ax)

axes[0].plot(list(K_range), inertias, 'o-', color=COLORS['cyan'], linewidth=2.5, markersize=8)
axes[0].axvline(x=best_k, color=COLORS['red'], linestyle='--', alpha=0.8, label=f'K={best_k}')
axes[0].set_title('Elbow Method', color='white', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Clusters (K)', color=THEME['accent'])
axes[0].set_ylabel('Inertia', color=THEME['accent'])
axes[0].legend(facecolor=THEME['card'], labelcolor='white')

axes[1].plot(list(K_range), silhouettes, 's-', color='#7CFC00', linewidth=2.5, markersize=8)
axes[1].axvline(x=best_k, color=COLORS['red'], linestyle='--', alpha=0.8, label=f'K={best_k}')
axes[1].set_title('Silhouette Score', color='white', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Clusters (K)', color=THEME['accent'])
axes[1].set_ylabel('Score', color=THEME['accent'])
axes[1].legend(facecolor=THEME['card'], labelcolor='white')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_elbow_silhouette.png', dpi=150, bbox_inches='tight', facecolor=THEME['dark'])
plt.close()
print("✓ 01_elbow_silhouette.png")

# ───────── PLOT 2: PCA SCATTER ─────────
fig, ax = plt.subplots(figsize=(13, 8), facecolor=THEME['dark'])
styled_ax(ax)

cluster_colors = [COLORS['gold'], COLORS['red'], COLORS['cyan']]
for i in range(best_k):
    mask = cluster_labels == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=cluster_colors[i], s=170, alpha=0.92,
               label=f'Cluster {i}: {label_names[i]}', edgecolors='white', linewidths=0.6, zorder=3)
    for j, name in enumerate(state_names[mask]):
        ax.annotate(name, (X_pca[mask][j, 0], X_pca[mask][j, 1]),
                    fontsize=7, color='#DDDDDD', alpha=0.9, xytext=(5, 4), textcoords='offset points')

ax.set_title('K-Means Clustering (K=3): US States by Drug Overdose Deaths',
             color='white', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel(f'PC1 - Overall Burden ({pca.explained_variance_ratio_[0]*100:.1f}%)',
              color=THEME['accent'], fontsize=11)
ax.set_ylabel(f'PC2 - Drug Mix ({pca.explained_variance_ratio_[1]*100:.1f}%)',
              color=THEME['accent'], fontsize=11)
ax.legend(facecolor=THEME['card'], labelcolor='white', fontsize=10, loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.08, color='white')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_clustering_pca.png', dpi=150, bbox_inches='tight', facecolor=THEME['dark'])
plt.close()
print("✓ 02_clustering_pca.png")

# ───────── PLOT 3: DRUG PROFILES ─────────
indicator_cols = BASE_INDICATORS
labels_short = ['Cocaine', 'Heroin', 'Methadone', 'Natural\nOpioids', 'Synthetic\nOpioids', 'Psycho-\nstimulants']

fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=THEME['dark'])
for i in range(best_k):
    ax = axes[i]
    ax.set_facecolor(THEME['card'])
    mask = cluster_labels == i
    states_in = state_names[mask]
    vals = [features[features['State Name'].isin(states_in)][col].mean() for col in indicator_cols]
    bars = ax.bar(labels_short, vals, color=cluster_colors[i], alpha=0.88, edgecolor='white', linewidth=0.5)
    ax.set_title(f'Cluster {i} - {label_names[i]}\n({len(states_in)} states | {cluster_chars[i]["deaths"]:.0f} deaths/mo)',
                 color='white', fontsize=10, fontweight='bold')
    ax.tick_params(colors=THEME['text'], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#333')
    ax.set_ylabel('Deaths/Month', color=THEME['accent'], fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}',
                ha='center', va='bottom', color='white', fontsize=8)

plt.suptitle('Drug Type Profile by Cluster (Actual Monthly Deaths)',
             color='white', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_cluster_profiles.png', dpi=150, bbox_inches='tight', facecolor=THEME['dark'])
plt.close()
print("✓ 03_cluster_profiles.png")

# ───────── PLOT 4: US GEOGRAPHICAL HEATMAP ─────────


state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'District of Columbia': 'DC',
}

mapdata = result_df.copy()
# Remove NYC (maps to NY — would conflict with New York state)
# Remove Puerto Rico (not a US state for choropleth scope='usa')
mapdata = mapdata[~mapdata['State'].isin(['New York City', 'Puerto Rico', 'United States'])]
mapdata['State_Code'] = mapdata['State'].map(state_abbrev)
mapdata = mapdata.dropna(subset=['State_Code', 'Avg Actual Deaths/Mo'])

print(f"\nMap coverage: {len(mapdata)} states")
print(mapdata[['State', 'State_Code', 'Avg Actual Deaths/Mo', 'Cluster Label']]
      .sort_values('Avg Actual Deaths/Mo', ascending=False).to_string())

hover_text = [
    f"<b>{mapdata.iloc[i]['State']}</b><br>"
    f"Cluster: {mapdata.iloc[i]['Cluster Label']}<br>"
    f"Avg Deaths/Mo: {mapdata.iloc[i]['Avg Actual Deaths/Mo']:.1f}<br>"
    f"Trend: {mapdata.iloc[i]['Trend (deaths/mo)']:+.3f} deaths/mo"
    for i in range(len(mapdata))
]

fig = go.Figure(data=go.Choropleth(
    locations=mapdata['State_Code'].values,
    z=mapdata['Avg Actual Deaths/Mo'].values,
    locationmode='USA-states',           # ← REQUIRED: tells plotly these are state codes
    text=hover_text,
    hovertemplate='%{text}<extra></extra>',
    colorscale=[
        [0.00, '#0a0a1a'],
        [0.10, '#1a237e'],
        [0.30, '#6a1b9a'],
        [0.55, '#c62828'],
        [0.75, '#ff6f00'],
        [1.00, '#ffeb3b'],
    ],
    marker_line_color='white',
    marker_line_width=1.0,
    colorbar=dict(
        thickness=20, len=0.75,
        title=dict(text='Avg Deaths/Month', font=dict(color='white', size=11)),
        tickfont=dict(color='white', size=10),
        bgcolor='rgba(26,26,46,0.9)',
        bordercolor='#555',
        borderwidth=1,
    )
))

fig.update_geos(
    scope='usa',
    projection_type='albers usa',
    showland=True,
    landcolor=THEME['card'],
    showlakes=True,
    lakecolor='#1a3a5c',
    coastlinecolor='#555',
    coastlinewidth=1,
    subunitcolor='#444',       # state borders
    subunitwidth=1,
    showsubunits=True,
    bgcolor=THEME['dark'],
)

fig.update_layout(
    title={
        'text': 'US Drug Overdose Deaths — Geographic Heatmap',
        'font': {'size': 20, 'color': 'white', 'family': 'Arial Black'},
        'x': 0.5, 'xanchor': 'center',
    },
    geo=dict(bgcolor=THEME['dark']),
    paper_bgcolor=THEME['dark'],
    plot_bgcolor=THEME['dark'],
    font=dict(color='white', size=12),
    height=700,
    margin=dict(l=0, r=0, t=80, b=0),
)

fig.write_html(f'{OUTPUT_DIR}/04_us_clustering_map.html')
print("✓ 04_us_clustering_map.html")

# ───────── PLOT 5: YEAR-WISE GEOGRAPHIC ANALYSIS ─────────
# Calculate deaths per year for each state
state_monthly['Year'] = state_monthly['Date'].dt.year
year_state_avg = state_monthly.groupby(['State Name', 'Year'])['Actual Monthly Deaths'].mean().reset_index()
year_state_avg.columns = ['State Name', 'Year', 'Avg Deaths/Mo']

# Get unique years and add cluster info
years = sorted(year_state_avg['Year'].unique())
year_state_avg = year_state_avg.merge(
    result_df[['State', 'Cluster', 'Cluster Label']],
    left_on='State Name', right_on='State', how='left'
).drop('State', axis=1)

# Build traces for each year
traces = []
for year in years:
    year_data = year_state_avg[year_state_avg['Year'] == year].copy()
    year_data['State_Code'] = year_data['State Name'].map(state_abbrev)
    year_data = year_data.dropna(subset=['State_Code', 'Avg Deaths/Mo'])
    
    hover_year = [
        f"<b>{year_data.iloc[i]['State Name']}</b><br>"
        f"Year: {year}<br>"
        f"Avg Deaths/Mo: {year_data.iloc[i]['Avg Deaths/Mo']:.1f}<br>"
        f"Cluster: {year_data.iloc[i]['Cluster Label']}"
        for i in range(len(year_data))
    ]
    
    trace = go.Choropleth(
        locations=year_data['State_Code'].values,
        z=year_data['Avg Deaths/Mo'].values,
        locationmode='USA-states',
        text=hover_year,
        hovertemplate='%{text}<extra></extra>',
        colorscale=[
            [0.00, '#0a0a1a'],
            [0.10, '#1a237e'],
            [0.30, '#6a1b9a'],
            [0.55, '#c62828'],
            [0.75, '#ff6f00'],
            [1.00, '#ffeb3b'],
        ],
        marker_line_color='white',
        marker_line_width=1.0,
        colorbar=dict(
            thickness=20, len=0.75,
            title=dict(text='Avg Deaths/Month', font=dict(color='white', size=11)),
            tickfont=dict(color='white', size=10),
            bgcolor='rgba(26,26,46,0.9)',
            bordercolor='#555',
            borderwidth=1,
            x=1.02
        ),
        name=str(year),
        visible=(year == years[0])  # Only first year visible by default
    )
    traces.append(trace)

# Create buttons for year selection
buttons = []
for i, year in enumerate(years):
    visible = [j == i for j in range(len(years))]
    buttons.append(
        dict(
            label=str(year),
            method='update',
            args=[{'visible': visible},
                  {'title': f'US Drug Overdose Deaths — Year-wise Geographic Analysis ({year})'}]
        )
    )

fig_year = go.Figure(data=traces)

fig_year.update_geos(
    scope='usa',
    projection_type='albers usa',
    showland=True,
    landcolor=THEME['card'],
    showlakes=True,
    lakecolor='#1a3a5c',
    coastlinecolor='#555',
    coastlinewidth=1,
    subunitcolor='#444',
    subunitwidth=1,
    showsubunits=True,
    bgcolor=THEME['dark'],
)

fig_year.update_layout(
    updatemenus=[
        dict(
            type='buttons',
            direction='left',
            x=0.05, y=1.15,
            buttons=buttons,
            bgcolor='rgba(26,26,46,0.9)',
            bordercolor='white',
            borderwidth=1,
            font=dict(color='white', size=10),
            active=0
        )
    ],
    title={
        'text': f'US Drug Overdose Deaths — Year-wise Geographic Analysis ({years[0]})',
        'font': {'size': 20, 'color': 'white', 'family': 'Arial Black'},
        'x': 0.5, 'xanchor': 'center',
    },
    geo=dict(bgcolor=THEME['dark']),
    paper_bgcolor=THEME['dark'],
    plot_bgcolor=THEME['dark'],
    font=dict(color='white', size=12),
    height=750,
    margin=dict(l=0, r=0, t=120, b=0),
)

fig_year.write_html(f'{OUTPUT_DIR}/05_year_analysis_map.html')
print("✓ 05_year_analysis_map.html")

# ───────── PLOT 6: YEAR-WISE TREND BY CLUSTER ─────────
year_cluster_avg = year_state_avg.groupby(['Year', 'Cluster Label'])['Avg Deaths/Mo'].mean().reset_index()

fig_trend, ax = plt.subplots(figsize=(14, 7), facecolor=THEME['dark'])
styled_ax(ax)

cluster_order = ['Low Volume / Rural', 'Moderate & Rising', 'High Burden Crisis']
colors_trend = [COLORS['gold'], COLORS['red'], COLORS['cyan']]

for cluster_name, color in zip(cluster_order, colors_trend):
    data = year_cluster_avg[year_cluster_avg['Cluster Label'] == cluster_name]
    ax.plot(data['Year'], data['Avg Deaths/Mo'], marker='o', linewidth=3, markersize=8,
            label=cluster_name, color=color, alpha=0.9)

ax.set_title('Year-wise Drug Overdose Deaths Trend by Cluster',
             color='white', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Year', color=THEME['accent'], fontsize=12)
ax.set_ylabel('Avg Actual Deaths/Month', color=THEME['accent'], fontsize=12)
ax.legend(facecolor=THEME['card'], labelcolor='white', fontsize=11, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.2, color='white', linestyle='--')
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_year_trend_by_cluster.png', dpi=150, bbox_inches='tight', facecolor=THEME['dark'])
plt.close()
print("✓ 06_year_trend_by_cluster.png\n")
print(f"All outputs saved to: {OUTPUT_DIR}")
print(f"\nData Files:")
print(f"  - cluster_assignments.csv (state-to-cluster mapping)")
print(f"\nModel Files:")
print(f"  - kmeans_model.pkl        (fitted KMeans + scaler + metadata)")
print(f"  - pca_model.pkl           (PCA transformer)")
print(f"\nVisualization Files:")
print(f"  - 01_elbow_silhouette.png (K selection metrics)")
print(f"  - 02_clustering_pca.png   (2D PCA scatter with state labels)")
print(f"  - 03_cluster_profiles.png (drug type breakdown per cluster)")
print(f"  - 04_us_clustering_map.html (interactive US geographical heatmap)")
print(f"  - 05_state_summary.png    (cluster summary with state counts)")
print(f"  - 06_year_trend_by_cluster.png (year-wise trend analysis)")
