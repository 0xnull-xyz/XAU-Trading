import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import misc

# Mute sklearn warnings
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

filename = "xau_df-1996.csv"
# df = investpy.get_currency_cross_historical_data(currency_cross="XAU/USD",
#                                             from_date='01/01/2018',
#                                             to_date='03/08/2021')
df = pd.read_csv(filename)
# df.set_axis(df['Date'], inplace=True)
df.set_index('Date', inplace=True)
df.index = pd.to_datetime(df.index)
df.drop(columns=['Currency'], inplace=True)

df.insert(0, 'Date', df.index)
df['Date'] = pd.to_datetime(df['Date'])

df_close = df[['Date', 'Close']].copy()
df_close = df_close.set_index('Date')
df_close.head()

df['EMA_9'] = df['Close'].ewm(9).mean().shift()

df = misc.applyFeatures(df)

# df['CloseTP1'] = df['Close']
# df['CloseTP1'].shift(-1)
df['Close'] = df['Close'].shift(-1)

df = df.iloc[51:]  # Because of moving averages and MACD line
df = df[:-1]  # Because of shifting close price
df = df.dropna()

df.index = range(len(df))

test_size = 0.15
valid_size = 0.15

test_split_idx = int(df.shape[0] * (1 - test_size))
valid_split_idx = int(df.shape[0] * (1 - (valid_size + test_size)))

train_df = df.loc[:valid_split_idx].copy()
valid_df = df.loc[valid_split_idx + 1:test_split_idx].copy()
test_df = df.loc[test_split_idx + 1:].copy()

drop_cols = ['Date', 'Open', 'Low', 'High']

train_df = train_df.drop(drop_cols, 1)
valid_df = valid_df.drop(drop_cols, 1)
test_df = test_df.drop(drop_cols, 1)

y_train = train_df['Close'].copy()
X_train = train_df.drop(['Close'], 1)

y_valid = valid_df['Close'].copy()
X_valid = valid_df.drop(['Close'], 1)

y_test = test_df['Close'].copy()
X_test = test_df.drop(['Close'], 1)

X_train.info()

parameters = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'max_depth': [8, 10, 12, 15],
    'gamma': [0.001, 0.005, 0.01, 0.02],
    'random_state': [42]
}

eval_set = [(X_train, y_train), (X_valid, y_valid)]
model = xgb.XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbose=False)
clf = GridSearchCV(model, parameters)

clf.fit(X_train, y_train, eval_set=[(X_train, y_train)])

print(f'Best params: {clf.best_params_}')
print(f'Best validation score = {clf.best_score_}')

model = xgb.XGBRegressor(**clf.best_params_, objective='reg:squarederror')
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

plot_importance(model)

y_pred = model.predict(X_test)
print(f'y_true = {np.array(y_test)[:5]}')
print(f'y_pred = {y_pred[:5]}')

print(f'mean_squared_error = {mean_squared_error(y_test, y_pred)}')
