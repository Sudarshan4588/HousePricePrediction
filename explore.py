import pandas as pd

df = pd.read_csv('House_Price.csv')
print(df.head())
print(df.columns)
print(df.info())
print(df.isnull().sum())
