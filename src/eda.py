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

if __name__ == "__main__":
    df = load_data()
    # structural_overview(df)
    indicator_overview(df)