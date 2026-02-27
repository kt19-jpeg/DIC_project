# Import necessary libraries
import pandas as pd
import numpy as np
import os

# Define file paths
raw_data_path = '/Users/kavyansh/DIC_project/data/raw/VSRR_Provisional_Drug_Overdose_Death_Counts_20260214.csv'
processed_data_path = '/Users/kavyansh/DIC_project/data/processed/cleaned_drug_overdose_deaths.csv'

# Read the raw data
df = pd.read_csv(raw_data_path)

# Basic data cleaning steps
# Drop rows with any missing values
df_cleaned = df.dropna()
# Remove duplicate rows
df_cleaned = df_cleaned.drop_duplicates()
# Optionally, reset index
df_cleaned = df_cleaned.reset_index(drop=True)

# Combine 'Year' and 'Month' to create a new 'Date' column and dropping unnecessary data
df_cleaned['Date'] = pd.to_datetime(df_cleaned['Year'].astype(str) + '-' + df_cleaned['Month'] + '-01')
df_cleaned.drop(['Year', 'Month','Period','Footnote','Footnote Symbol'], axis=1, inplace=True)

df_cleaned_filtered = df_cleaned[(df_cleaned['Percent Complete'] == 100) & (df_cleaned['Percent Pending Investigation']< 0.3)]

# data index fixing
columns = df_cleaned_filtered.columns.tolist()
columns.remove('Date')
indicator_index = columns.index('Indicator')
columns.insert(indicator_index + 1, 'Date')
df_cleaned_filtered = df_cleaned_filtered[columns]
df_cleaned_filtered = df_cleaned_filtered.rename(columns={'Data Value': 'Death Count'})



# Ensure processed directory exists
os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

# Save the cleaned and filtered DataFrame to CSV
df_cleaned_filtered.to_csv(processed_data_path, index=False)

# Save the cleaned data
df_cleaned_filtered.to_csv(processed_data_path, index=False)

print(f"Cleaned data saved to {processed_data_path}")