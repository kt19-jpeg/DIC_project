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
data_path = Path(__file__).parent.parent / 'data' / 'VSRR_Provisional_Drug_Overdose_Death_Counts_20260214.csv'
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
