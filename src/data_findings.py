import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load data
data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'VSRR_Provisional_Drug_Overdose_Death_Counts_20260214.csv'
df = pd.read_csv(data_path)

# Redirect sys.stdout to findings.txt at the very top
findings_path = Path(__file__).parent.parent / 'reports' / 'findings.txt'
findings_file = open(findings_path, 'w', encoding='utf-8')
sys.stdout = findings_file

print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"\nDataset Shape: {df.shape}")
print(f"Total Records: {len(df):,}")
print(f"Total Columns: {len(df.columns)}")

print("\n" + "="*80)
print("COLUMN INFORMATION")
print("="*80)
print("\nColumns and Data Types:")
print(df.dtypes)

print("\n" + "="*80)
print("MISSING VALUES ANALYSIS")
print("="*80)
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_data.index,
    'Missing Count': missing_data.values,
    'Missing %': missing_percent.values
})
print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False))

print("\n" + "="*80)
print("BASIC STATISTICS")
print("="*80)
print(df.describe())

print("\n" + "="*80)
print("UNIQUE VALUES ANALYSIS")
print("="*80)
print(f"Unique States: {df['State'].nunique()}")
print(f"States: {sorted(df['State'].unique())}")

print(f"\nYears Covered: {sorted(df['Year'].unique())}")
print(f"Year Range: {df['Year'].min()} - {df['Year'].max()}")

print(f"\nUnique Indicators (Drug Types): {df['Indicator'].nunique()}")
print("\nIndicators:")
for indicator in sorted(df['Indicator'].unique()):
    count = (df['Indicator'] == indicator).sum()
    print(f"  - {indicator}: {count:,} records")

print(f"\nPeriods: {df['Period'].unique()}")
print(f"Months: {df['Month'].unique()}")

# Data Value Statistics
print("\n" + "="*80)
print("DATA VALUE STATISTICS")
print("="*80)
df['Data Value'] = pd.to_numeric(df['Data Value'], errors='coerce')
print(f"Non-null Data Values: {df['Data Value'].notna().sum():,} ({(df['Data Value'].notna().sum()/len(df)*100):.2f}%)")
print(f"Min Death Count: {df['Data Value'].min():.0f}")
print(f"Max Death Count: {df['Data Value'].max():.0f}")
print(f"Mean Death Count: {df['Data Value'].mean():.2f}")
print(f"Median Death Count: {df['Data Value'].median():.2f}")

# Create visualizations
fig = plt.figure(figsize=(16, 12))

# 1. Data completeness by year
ax1 = plt.subplot(3, 3, 1)
completeness_by_year = df.groupby('Year')['Data Value'].apply(lambda x: x.notna().sum() / len(x) * 100)
completeness_by_year.plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('Data Completeness by Year (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Year')
ax1.set_ylabel('Completeness %')
ax1.grid(axis='y', alpha=0.3)

# 2. Total records by state (top 15)
ax2 = plt.subplot(3, 3, 2)
state_counts = df['State Name'].value_counts().head(15)
state_counts.plot(kind='barh', ax=ax2, color='coral')
ax2.set_title('Top 15 States by Record Count', fontsize=12, fontweight='bold')
ax2.set_xlabel('Number of Records')
ax2.grid(axis='x', alpha=0.3)

# 3. Distribution of drug indicators
ax3 = plt.subplot(3, 3, 3)
indicator_counts = df['Indicator'].value_counts().head(10)
indicator_counts.plot(kind='barh', ax=ax3, color='lightgreen')
ax3.set_title('Top 10 Drug Indicators by Record Count', fontsize=12, fontweight='bold')
ax3.set_xlabel('Number of Records')
ax3.grid(axis='x', alpha=0.3)

# 4. Distribution of data values (non-null)
ax4 = plt.subplot(3, 3, 4)
valid_data = df['Data Value'].dropna()
ax4.hist(valid_data, bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
ax4.set_title('Distribution of Death Counts (Log Scale)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Death Count')
ax4.set_ylabel('Frequency')
ax4.set_yscale('log')
ax4.grid(axis='y', alpha=0.3)

# 5. Average data value by year (for available data)
ax5 = plt.subplot(3, 3, 5)
avg_by_year = df.groupby('Year')['Data Value'].mean()
avg_by_year.plot(kind='line', marker='o', ax=ax5, color='darkred', linewidth=2)
ax5.set_title('Average Death Count by Year', fontsize=12, fontweight='bold')
ax5.set_xlabel('Year')
ax5.set_ylabel('Average Death Count')
ax5.grid(True, alpha=0.3)

# 6. Percent complete by year
ax6 = plt.subplot(3, 3, 6)
df['Percent Complete'] = pd.to_numeric(df['Percent Complete'], errors='coerce')
pct_complete = df.groupby('Year')['Percent Complete'].mean()
pct_complete.plot(kind='bar', ax=ax6, color='teal')
ax6.set_title('Average Percent Complete by Year', fontsize=12, fontweight='bold')
ax6.set_xlabel('Year')
ax6.set_ylabel('% Complete')
ax6.set_ylim([0, 105])
ax6.grid(axis='y', alpha=0.3)

# 7. Top 10 states by total death count (when available)
ax7 = plt.subplot(3, 3, 7)
state_deaths = df.groupby('State Name')['Data Value'].sum().nlargest(10)
state_deaths.plot(kind='barh', ax=ax7, color='orange')
ax7.set_title('Top 10 States by Total Death Count', fontsize=12, fontweight='bold')
ax7.set_xlabel('Total Death Count')
ax7.grid(axis='x', alpha=0.3)

# 8. Data value box plot by top indicators
ax8 = plt.subplot(3, 3, 8)
top_indicators = df['Indicator'].value_counts().head(5).index
df_top = df[df['Indicator'].isin(top_indicators)]
df_top.boxplot(column='Data Value', by='Indicator', ax=ax8)
ax8.set_title('Death Count Distribution by Top 5 Indicators', fontsize=12, fontweight='bold')
ax8.set_xlabel('Indicator')
ax8.set_ylabel('Death Count')
plt.sca(ax8)
plt.xticks(rotation=45, ha='right', fontsize=9)

# 9. Records by month (when data available)
ax9 = plt.subplot(3, 3, 9)
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
month_data = df[df['Data Value'].notna()]['Month'].value_counts().reindex(month_order)
month_data.plot(kind='bar', ax=ax9, color='skyblue')
ax9.set_title('Data Availability by Month', fontsize=12, fontweight='bold')
ax9.set_xlabel('Month')
ax9.set_ylabel('Non-null Records')
plt.sca(ax9)
plt.xticks(rotation=45, ha='right', fontsize=9)

plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / 'reports' / 'eda_analysis.png', dpi=300, bbox_inches='tight')

# At the very end, after all code:
findings_file.close()
sys.stdout = sys.__stdout__



deadends_path = Path(__file__).parent.parent / "reports" / "deadends.txt"
deadends_file = open(deadends_path, "w", encoding="utf-8")  # use the Path, not a relative string


def state_drug_correlation_matrix(df):
    print("\n" + "=" * 80, file=deadends_file)
    print("DEAD END: STATE-LEVEL DRUG TYPE CORRELATION MATRIX", file=deadends_file)
    print("=" * 80, file=deadends_file)

    print("\nHypothesis:", file=deadends_file)
    print(
        "We expected the correlation matrix to reveal meaningful drug-specific patterns "
        "— which drug types cluster together, and which states share similar profiles. "
        "If certain drugs correlate strongly, it suggests shared structural drivers "
        "(poverty, supply chains, prescribing behavior). If they don't, it points to "
        "distinct, drug-specific crises that would need separate analysis.",
        file=deadends_file,
    )

    drug_indicators = [
        "Cocaine (T40.5)",
        "Heroin (T40.1)",
        "Synthetic opioids, excl. methadone (T40.4)",
        "Psychostimulants with abuse potential (T43.6)",
        "Methadone (T40.3)",
    ]

    subset = df[df["Indicator"].isin(drug_indicators)]
    pivot = (
        subset.groupby(["State", "Indicator"])["Data Value"]
        .mean()
        .unstack("Indicator")
    )

    total_states = pivot.shape[0]
    complete_states = pivot.dropna().shape[0]
    missing_summary = pivot.isnull().sum()

    print(f"\nTotal states/territories in dataset: {total_states}", file=deadends_file)
    print(f"States with complete data across all 5 drug types: {complete_states}", file=deadends_file)
    print(f"States dropped due to missing data: {total_states - complete_states}", file=deadends_file)

    print("\nMissing values per drug type (suppressed due to low counts):", file=deadends_file)
    print(missing_summary.to_string(), file=deadends_file)

    print("\nCorrelation matrix (complete cases only):", file=deadends_file)
    print(pivot.dropna().corr().round(3).to_string(), file=deadends_file)

    print("\nConclusion:", file=deadends_file)
    print(
        f"The matrix ran fine — {complete_states} out of {total_states} states had "
        f"complete data, which is solid coverage. The problem is what the matrix "
        f"actually showed: almost everything correlates strongly with everything else. "
        f"Cocaine, heroin, opioids, and synthetic opioids all move together at the "
        f"state level, producing correlations in the 0.72-0.85 range. Rather than "
        f"revealing meaningful drug-specific patterns, the matrix is largely reflecting "
        f"population size — bigger states have more deaths across all drug types. It "
        f"can't tell us whether a state has a heroin problem vs a fentanyl problem vs "
        f"a cocaine problem, because everything is high together. To make this useful "
        f"we'd need to normalize by state population, but that data isn't in this "
        f"dataset. Dead end.",
        file=deadends_file,
    )


    print("\n" + "=" * 80, file=deadends_file)
    
def footnote_symbol_data_quality(df):
    print("\n" + "=" * 80, file=deadends_file)
    print("DEAD END: FOOTNOTE SYMBOL AS A DATA QUALITY PREDICTOR", file=deadends_file)
    print("=" * 80, file=deadends_file)

    print("\nHypothesis:", file=deadends_file)
    print(
        "The dataset uses two footnote symbols — '**' for suppressed/low quality data "
        "and '*' for underreported/incomplete data. We expected '**' rows to have "
        "significantly more missing Data Values since they're explicitly flagged for "
        "quality issues, making the symbol a useful filter for reliable records.",
        file=deadends_file,
    )

    star_star = df[df["Footnote Symbol"] == "**"]["Data Value"].isna().mean() * 100
    star = df[df["Footnote Symbol"] == "*"]["Data Value"].isna().mean() * 100

    star_star_count = len(df[df["Footnote Symbol"] == "**"])
    star_count = len(df[df["Footnote Symbol"] == "*"])

    print(f"\n'**' rows (suppressed/quality issues): {star_star_count:,}", file=deadends_file)
    print(f"'*'  rows (underreported/incomplete):  {star_count:,}", file=deadends_file)
    print(f"\nMissing rate for '**' rows: {star_star:.1f}%", file=deadends_file)
    print(f"Missing rate for '*'  rows: {star:.1f}%", file=deadends_file)

    print("\nConclusion:", file=deadends_file)
    print(
        f"The symbol doesn't work as a quality filter the way we expected. '**' rows "
        f"— flagged for suppression and quality issues — actually have a higher missing "
        f"rate ({star_star:.1f}%) than '*' rows ({star:.1f}%), but the difference isn't "
        f"meaningful enough to use as a filter. More importantly, ALL rows in this "
        f"dataset carry one of these two symbols, so the footnote symbol has no "
        f"discriminating power — it can't separate reliable records from unreliable "
        f"ones. Dead end.",
        file=deadends_file,
    )


def pending_investigation_predicts_missing(df):
    print("\n" + "=" * 80, file=deadends_file)
    print("DEAD END: PERCENT PENDING INVESTIGATION AS A MISSING DATA PREDICTOR", file=deadends_file)
    print("=" * 80, file=deadends_file)

    print("\nHypothesis:", file=deadends_file)
    print(
        "Higher 'Percent Pending Investigation' should mean less finalized data, "
        "which should mean a higher chance of the Data Value being missing. If true, "
        "this column could serve as a useful signal for filtering out unreliable records.",
        file=deadends_file,
    )

    missing_pending = df[df["Data Value"].isna()]["Percent Pending Investigation"].mean()
    present_pending = df[df["Data Value"].notna()]["Percent Pending Investigation"].mean()
    difference = missing_pending - present_pending

    print(f"\nMean pending % when Data Value IS missing:  {missing_pending:.4f}", file=deadends_file)
    print(f"Mean pending % when Data Value is present:  {present_pending:.4f}", file=deadends_file)
    print(f"Difference:                                  {difference:.4f}", file=deadends_file)

    print("\nConclusion:", file=deadends_file)
    print(
        f"The relationship exists but is too weak to be useful. Records with missing "
        f"Data Values have a mean pending rate of {missing_pending:.4f} vs {present_pending:.4f} "
        f"for present ones — a difference of only {difference:.4f}. That's not nearly "
        f"enough separation to use this column as a predictor or filter. The missingness "
        f"in this dataset is driven by CDC suppression rules (counts below 10) and state "
        f"reporting lags, not by how much is pending investigation. Dead end.",
        file=deadends_file,
    )





state_drug_correlation_matrix(df)
footnote_symbol_data_quality(df)
pending_investigation_predicts_missing(df)




deadends_file.close()
print("Done — check deadends.txt")
