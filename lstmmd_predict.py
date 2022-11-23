import datetime
import itertools
from datetime import timedelta
from pickle import load
import socket

import investpy
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import keras
import requests
import tensorflow as tf
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler

import misc

close_col_idx = 1


# close_col_idx = 11


# detect and init the TPU
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

api_id = '46169'
api_hash = '06cb96ba47abd82be83ae5713b273ccc'
token = '1946485856:AAHafVNwkJ7f9JEzcTxnr1xLZ_99HtzXNuM'
chat_id = '-519080911'
message = "Working..."
# your phone number
phone = '+989109297642'
look_back = 13
last_prediction = ''


def telegram_bot_sendtext(bot_message):
    bot_token = token
    bot_chatID = chat_id
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    response = requests.get(send_text)

    return response.json()



class socketserver:
    def __init__(self, address='', port=9090):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.address = address
        self.port = port
        self.sock.bind((self.address, self.port))
        self.cummdata = ''

    def recvmsg(self):
        self.sock.listen(100)
        self.conn, self.addr = self.sock.accept()
        print('connected to', self.addr)
        self.cummdata = ''

        while True:
            data = self.conn.recv(10000)
            self.cummdata += data.decode("utf-8")
            if not data:
                break
            with open('tt.txt', 'rb') as reader:
                tt = reader.readlines()
            first_line = tt[0].split(sep=b',')
            response = first_line[1]
            if first_line[0].decode().lower().startswith(self.cummdata):
                self.conn.send(response)
            else:
                self.conn.send(b'')
            return self.cummdata

    def __del__(self):
        self.sock.close()


def LSTM_HyperParameter_Tuning(config, x_train, y_train, x_test, y_test):
    first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout = config
    possible_combinations = list(
        itertools.product(first_additional_layer, second_additional_layer, third_additional_layer,
                          n_neurons, n_batch_size, dropout))

    print(possible_combinations)
    print('\n')

    hist = []

    for i in range(0, len(possible_combinations)):

        print(f'{i + 1}th combination: \n')
        print('--------------------------------------------------------------------')

        first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout = \
            possible_combinations[i]

        # instantiating the model in the strategy scope creates the model on the TPU
        # with tpu_strategy.scope():
        regressor = Sequential()
        regressor.add(LSTM(units=n_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        regressor.add(Dropout(dropout))

        if first_additional_layer:
            regressor.add(LSTM(units=n_neurons, return_sequences=True))
            regressor.add(Dropout(dropout))

        if second_additional_layer:
            regressor.add(LSTM(units=n_neurons, return_sequences=True))
            regressor.add(Dropout(dropout))

        if third_additional_layer:
            regressor.add(GRU(units=n_neurons, return_sequences=True))
            regressor.add(Dropout(dropout))

        regressor.add(LSTM(units=n_neurons, return_sequences=False))
        regressor.add(Dropout(dropout))
        regressor.add(Dense(units=1, activation='linear'))
        regressor.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        '''''
        From the mentioned article above --> If a validation dataset is specified to the fit() function via the validation_data or v
        alidation_split arguments,then the loss on the validation dataset will be made available via the name “val_loss.”
        '''''

        file_path = 'best_model.h5'

        mc = ModelCheckpoint(file_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

        '''''
        cb = Callback(...)  # First, callbacks must be instantiated.
        cb_list = [cb, ...]  # Then, one or more callbacks that you intend to use must be added to a Python list.
        model.fit(..., callbacks=cb_list)  # Finally, the list of callbacks is provided to the callback argument when fitting the model.
        '''''

        regressor.fit(x_train, y_train, validation_split=0.3, epochs=40, batch_size=n_batch_size, callbacks=[es, mc],
                      verbose=0)

        # load the best model
        # regressor = load_model('best_model.h5')

        train_accuracy = regressor.evaluate(x_train, y_train, verbose=0)
        test_accuracy = regressor.evaluate(x_test, y_test, verbose=0)

        hist.append(list(
            (first_additional_layer, second_additional_layer, third_additional_layer, n_neurons, n_batch_size, dropout,
             train_accuracy, test_accuracy)))

        print(
            f'{str(i)}-th combination = {possible_combinations[i]} \n train accuracy: {train_accuracy} and test accuracy: {test_accuracy}')

        print('--------------------------------------------------------------------')
        print('--------------------------------------------------------------------')
        print('--------------------------------------------------------------------')
        print('--------------------------------------------------------------------')

    return hist


# Creating a data structure (it does not work when you have only one feature)
def create_data(df, n_future, n_past, train_test_split_percentage, validation_split_percentage):
    n_feature = df.shape[1]
    x_data, y_data = [], []

    for i in range(n_past, len(df) - n_future + 1):
        x_data.append(df.iloc[i - n_past:i, 0:n_feature])
        y_data.append(df.iloc[i + n_future - 1:i + n_future, close_col_idx])

    split_training_test_starting_point = int(round(train_test_split_percentage * len(x_data)))
    split_train_validation_starting_point = int(
        round(split_training_test_starting_point * (1 - validation_split_percentage)))

    x_train = x_data[:split_train_validation_starting_point]
    y_train = y_data[:split_train_validation_starting_point]

    # if you want to choose the validation set by yourself, uncomment the below code.
    x_val = x_data[split_train_validation_starting_point:split_training_test_starting_point]
    y_val = x_data[split_train_validation_starting_point:split_training_test_starting_point]

    x_test = x_data[split_training_test_starting_point:]
    y_test = y_data[split_training_test_starting_point:]

    return np.array(x_train), np.array(x_test), np.array(x_val), np.array(y_train), np.array(y_test), np.array(y_val)


# Creating a data structure (it does not work when you have only one feature)
def create_data_scaled(df, n_future, n_past, train_test_split_percentage, validation_split_percentage):
    n_feature = df.shape[1]
    x_data, y_data = [], []

    for i in range(n_past, len(df) - n_future + 1):
        x_data.append(df[i - n_past:i, 0:n_feature])
        y_data.append(df[i + n_future - 1:i + n_future, close_col_idx])

    split_training_test_starting_point = int(round(train_test_split_percentage * len(x_data)))
    split_train_validation_starting_point = int(
        round(split_training_test_starting_point * (1 - validation_split_percentage)))

    x_train = x_data[:split_train_validation_starting_point]
    y_train = y_data[:split_train_validation_starting_point]

    # if you want to choose the validation set by yourself, uncomment the below code.
    x_val = x_data[split_train_validation_starting_point:split_training_test_starting_point]
    y_val = x_data[split_train_validation_starting_point:split_training_test_starting_point]

    x_test = x_data[split_training_test_starting_point:]
    y_test = y_data[split_training_test_starting_point:]

    return np.array(x_train), np.array(x_test), np.array(x_val), np.array(y_train), np.array(y_test), np.array(y_val)


def create_data_predict(df, n_future, n_past, train_test_split_percentage, validation_split_percentage):
    n_feature = df.shape[1]
    x_data, y_data = [], []

    for i in range(n_past, len(df) - n_future + 1):
        x_data.append(df[i - n_past:i, 0:n_feature])
        # y_data.append(df[i + n_future - 1:i + n_future, close_col_idx])

    return np.array(x_data)


def create_data_predict_2(df, n_future, n_past, train_test_split_percentage, validation_split_percentage):
    n_feature = df.shape[1]
    x_data, y_data = [], []

    for i in range(n_past, len(df) + 1):
        x_data.append(df[i - n_past:i, 0:n_feature])
        # y_data.append(df[i + n_future - 1:i + n_future, close_col_idx])

    return np.array(x_data)


def ReportPrice():
    end = (datetime.datetime.now() - timedelta(days = 1)).strftime("%d/%m/%Y")
    df = investpy.get_currency_cross_historical_data(currency_cross="XAU/USD",
                                                     from_date='01/06/2021',
                                                     to_date=end)

    # filename = "xau_df-1996.csv"
    # rs_df = pd.read_csv(filename)
    df.index = pd.to_datetime(df.index)
    scaler = load(open('scaler.pkl', 'rb'))
    dsc = load(open('dsc.pkl', 'rb'))

    df.insert(0, 'Date', df.index)
    # df.set_index('Date', inplace=True)
    # df['Date'] = pd.to_datetime(df['Date'])
    df.set_axis(df['Date'], inplace=True)
    # df.drop(columns=['Open', 'High', 'Low', 'Currency'], inplace=True)
    df.drop(columns=['Currency'], inplace=True)

    df['MID'] = (df['High'] + df['Low']) / 2
    data_prices = df.drop(['Open', 'Low', 'High'], axis=1)
    # df['EMA_3'] = df['Close'].ewm(3).mean().shift()
    df['EMA_9'] = df['Close'].ewm(9).mean().shift()
    df = misc.rsi(df, 14, False)
    df = misc.moving_average(df, 5)
    # df = misc.bb(df, 50)
    df['Bias_5'] = (df['Close'] - df["MA_5"]) / df["MA_5"]
    df = df.drop('MA_5', axis=1)
    # df = df.iloc[51:]  # Because of moving averages and MACD line and bollinger
    # df = df[:-1]  # Because of shifting close price
    df = df.dropna()
    data_prices = df.drop('Date', axis=1)[-31:]

    # Feature Scaling
    # sc = MinMaxScaler(feature_range=(0, 1))
    sc = RobustScaler()
    Y_sc = RobustScaler()
    data_prices_scaled = dsc.transform(data_prices)
    # Y_sc.fit(data_prices['Close'].values.reshape(-1, 1))

    # Number of days you want to predict into the future
    # Number of past days you want to use to predict the future

    X_W_P = create_data_predict_2(data_prices_scaled, n_future=1, n_past=30,
                                  train_test_split_percentage=0.9,
                                  validation_split_percentage=0)
    # ------------------LSTM-----------------------
    regressor = Sequential()
    regressor.add(LSTM(units=16, return_sequences=False, activation='relu', input_shape=(X_W_P.shape[1], X_W_P.shape[2])))
    # regressor.add(Dropout(0.1))
    #
    # regressor.add(LSTM(units=16, activation='relu', return_sequences=False))
    # regressor.add(Dropout(0.1))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    print(regressor.summary())

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=35)

    # history = regressor.fit(X_train, y_train, validation_split=0.3, epochs=32, batch_size=32, callbacks=[es])
    # print(history.history.keys())

    regressor.load_weights('lstmi-x-2.json')
    y_pred = regressor.predict(X_W_P)
    # y_pred = Y_sc.inverse_transform(y_pred)
    # y_test = Y_sc.inverse_transform(y_test)
    print(df.index[-1] + timedelta(days=1))
    dir = 'N'
    if y_pred[0] > y_pred[1]:
        dir = 'down'
        print('down')
    if y_pred[1] > y_pred[0]:
        dir = 'up'
        print('up')
    with open('tt.txt', 'w') as writer:
        writer.write(f"{str(datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))},{dir}")
    # np.savetxt("predtest-lstmi-v.csv", y_pred, delimiter=",")
    # np.savetxt("realtest-lstmi-v.csv", data_prices, delimiter=",")
    # df.to_csv('rr.csv')


if __name__ == '__main__':
    ReportPrice()
    scheduler = BackgroundScheduler()
    scheduler.add_job(ReportPrice, 'cron', hour='1,2', minute='1,29,30,31,32,33,34,35,36,45,59')

    print("Scheduler Started!")
    scheduler.start()
    serv = socketserver('127.0.0.1', 9099)

    while True:
        msg = serv.recvmsg()