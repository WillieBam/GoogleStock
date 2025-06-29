import pandas as pd
import numpy as np
import pandas_market_calendars as cal
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
def test_train_split(df,train_ratio=0.7,val_ratio=0.15):
    total_len = len(df)
    train_end = int(total_len*train_ratio)
    return df.iloc[:train_end], df.iloc[train_end:]
    
    
    
def split_data(df, train_ratio=0.7,val_ratio=0.15):
    total_len = len(df)
    train_end = int(total_len*train_ratio)
    val_end = int(total_len*(val_ratio + train_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

def hl_range_features(df):
    df = df.copy()
    df['hl_range'] = df['High'] - df['Low']
    df['hl_range_ma'] = df['hl_range'].rolling(20).mean()
    df['hl_range_pct'] = df['hl_range']/df['Close']
    
    return df

def open_close_range_feature(df):
    df = df.copy()
    df['open_close'] = df['Close'] - df['Open']
    return df

def volume_spike(df):
    df=df.copy()
    df['volume_spike_ratio'] = df['Volume']/df['Volume'].rolling(20).mean()
    return df


def rolling_volatility(df, window=5):
    df= df.copy()
    df['roll_vola_price'] = df['Close'].rolling(window).std()
    # log return: continuous compounded returns -> measure investment performance using log
    # price at the end of the period /  price at the begining of the period
    df['log_return'] = np.log(df['Close']/df['Close'].shift(1)) 
    df['roll_vola_return'] = df['log_return'].rolling(window).std()
    
    return df

def clean_data(df):
    features_cols_to_be_cleansed=[ 'Date', 'Close', 'High', 'Low', 'Open', 'Volume',
    'Year', 'Month','YearMonth', 'label']
    df = df.dropna(subset=features_cols_to_be_cleansed)
    return df

def check_holidays():
    newyork_holiday = cal.get_calendar('XNYS')
    schedule = newyork_holiday.schedule(start_date='2020-06-04',end_date='2025-06-02')
    trade_days = schedule.index
    full_date_range = pd.date_range(start='2020-06-04',end='2025-06-02', freq='B')
    actual_holidays = set(full_date_range) - set(trade_days)
    return actual_holidays

def metrics_performance(y_true,y_pred,name):
    mae = mean_absolute_error(y_true,y_pred)
    rmse = mean_squared_error(y_true,y_pred)
    r2 = r2_score(y_true,y_pred)
    print(f'{name} - MAE:{mae:.4f}, RMSE:{rmse:.4f}, R2:{r2:.4f}')

"""
what is next:
minmax scaler, standard scaler
predict what
build x and y
build baseline model (LSTM, prophet, RandomForest, Linear Regression)

"""
