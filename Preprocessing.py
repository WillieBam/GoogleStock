import pandas as pd
import Pipeline as pipe
import plotly 

data = pd.read_csv('google_5yr_one.csv')
df = pd.DataFrame(data)

# EDA
print(df.columns)
print(df.describe())
header = df.columns
print(plotly.__version__)

# change type
for i in header[1:]:
  df[i] = pd.to_numeric(df[i], errors='coerce')
print(df.dtypes)

# data splitting
train_df, val_df, test_df = pipe.split_data(df)

# data preprocessing

