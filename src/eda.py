import pandas as pd
import os

DATA_PATH = "data/raw/U.S._Chronic_Disease_Indicators.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def structural_overview(df):
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)
    print("\nSample Rows:")
    print(df.head())

def indicator_overview(df):
    print("Unique Topics:", df['Topic'].nunique())
    print("Unique Questions:", df['Question'].nunique())
    print("\nTop 10 Topics:")
    print(df['Topic'].value_counts().head(10))

def missingness_summary(df):
    missing = df.isnull().sum().sort_values(ascending=False)
    percent_missing = (missing / len(df) * 100).round(2)
    
    summary = pd.DataFrame({
        "Missing Count": missing,
        "Percent Missing": percent_missing
    })
    
    print(summary)

def topic_subset_summary(df):
    subset = df[df['Topic'].isin([
        "Cardiovascular Disease",
        "Chronic Obstructive Pulmonary Disease"
    ])]
    
    print("Subset shape:", subset.shape)
    print("Missing DataValue in subset:", subset['DataValue'].isnull().sum())
    print("Percent missing in subset:",
          round(subset['DataValue'].isnull().mean() * 100, 2))
    
def subset_data_value_summary(df):
    subset = df[df['Topic'].isin([
        "Cardiovascular Disease",
        "Chronic Obstructive Pulmonary Disease"
    ])]
    
    clean_subset = subset.dropna(subset=['DataValue'])

    subset = df[df['Topic'].isin([
    "Cardiovascular Disease",
    "Chronic Obstructive Pulmonary Disease"
])]

    # print(subset['DataValueUnit'].value_counts())
    
    # print(clean_subset['DataValue'].describe())
    percent_subset = subset[subset['DataValueUnit'] == "%"]
    print(percent_subset['DataValue'].describe())

if __name__ == "__main__":
    df = load_data()
    # structural_overview(df)
    # indicator_overview(df)
    # missingness_summary(df)
    # topic_subset_summary(df)
    subset_data_value_summary(df)