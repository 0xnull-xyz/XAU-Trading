# GOLD(XAU/USD) Price Prediction Using LSTM
Daily close Prediction of gold Using LSTM & XGB.

# Dataset:
The dataset is taken from yahoo finace's website in CSV format. The dataset consists of Open, High, Low and Closing Prices of gold from 2nd january 1996 to 30th July 2021 - total 6660 rows. 
# Price Indicator:
We added some famous technical indicators such as RSI, ROC, Bollinger Bands, etc.

# Data Pre-processing:
After converting the dataset into daily close, it becomes one column data. This has been converted into multi column time series data. All values have been normalized between 0 and 1.
# Model: 
Two/Three sequential LSTM layers have been stacked together and one dense layer is used to build the RNN model using Keras deep learning library. Since this is a regression task, 'linear' activation has been used in final layer.
# Version:
Python 3.7 and latest versions of all libraries including deep learning library Keras and Tensorflow.
# Training:
75% data is used for training. Adagrad (adaptive gradient algorithm) optimizer is used for faster convergence.

# Test:
Test accuracy metric is root mean square error (RMSE).

# HyperParam Tuning:
Go to lstm.py

# Daily activation & send signal:
In lstm_predict.py a bot is created when everynight we try to predict tomorrow daily close & the signal will be sent to a telegram channel.