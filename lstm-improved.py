import itertools
from pickle import dump

import pandas as pd
import plotly.graph_objects as go
import numpy as np
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler

import misc

close_col_idx = 3


# close_col_idx = 11


# detect and init the TPU
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


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
def create_data_scaled(df, n_future, n_past, train_test_split_percentage, validation_split_percentage, raw):
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

    test_df = raw[split_training_test_starting_point + n_past:]

    return np.array(x_train), np.array(x_test), np.array(x_val), np.array(y_train), np.array(y_test), np.array(y_val), \
           test_df


filename = "xau_df-1996.csv"
df = pd.read_csv(filename)
print(df.info())

df['Date'] = pd.to_datetime(df['Date'])
df.set_axis(df['Date'], inplace=True)
# df.drop(columns=['Open', 'High', 'Low', 'Currency'], inplace=True)
df.drop(columns=['Currency'], inplace=True)

# df['MID'] = (df['High'] + df['Low']) / 2
# data_prices = df.drop(['Open', 'Low', 'High'], axis=1)
# df['EMA_3'] = df['Close'].ewm(3).mean().shift()
df['EMA_9'] = df['Close'].ewm(9).mean().shift()
df = misc.rsi(df, 14, False)
# df = misc.commodity_channel_index(df, 14)
# df = misc.commodity_channel_index(df, 24)
# df = misc.moving_average(df, 5)
df = misc.momentum(df, 14)
# df['alpha_054'] = - (df['Low'] - df['Close']) * df['Open'] ** 5 / ((df['Low'] - df['High']) * df['Close'] ** 5)
df = misc.bb(df, 50)
# df['Bias_5'] = (df['Close'] - df["MA_5"]) / df["MA_5"]
# df = df.drop('MA_5', axis=1)
df = df.iloc[51:]
pd.options.mode.use_inf_as_na = True
df = df.dropna()
data_prices = df.drop('Date', axis=1)

# Feature Scaling
# sc = MinMaxScaler(feature_range=(0, 1))
sc = RobustScaler()
Y_sc = RobustScaler()
data_prices_scaled = sc.fit_transform(data_prices)
Y_sc.fit(data_prices['Close'].values.reshape(-1, 1))

# Number of days you want to predict into the future
# Number of past days you want to use to predict the future

X_train, X_test, X_val, y_train, y_test, y_val, test_dates = create_data_scaled(data_prices_scaled, n_future=1,
                                                                                n_past=30,
                                                                                train_test_split_percentage=0.9,
                                                                                validation_split_percentage=0, raw=df)
# ------------------LSTM-----------------------
while True:
    regressor = Sequential()
    regressor.add(
        LSTM(units=32, return_sequences=False, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    # regressor.add(Dropout(0.1))
    #
    # regressor.add(LSTM(units=32, activation='relu', return_sequences=False))
    # regressor.add(Dropout(0.1))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    print(regressor.summary())

    es = EarlyStopping(monitor='root_mean_squared_error', mode='min', verbose=1, patience=128)
    history = regressor.fit(X_train, y_train, validation_split=0.2, epochs=32, batch_size=32, callbacks=[es])
    print(history.history.keys())

    regressor.save_weights('lstm-nw-1.json')
    dump(Y_sc, open('Y_scaler.pkl', 'wb'))
    dump(sc, open('dsc.pkl', 'wb'))
    # dump(regressor, open('model.pkl', 'wb'))

    y_pred = regressor.predict(X_test)
    y_pred = Y_sc.inverse_transform(y_pred)
    y_test = Y_sc.inverse_transform(y_test)

    test_dates['Real'] = y_test.reshape(1, -1)[0]
    test_dates['Pred'] = y_pred.reshape(1, -1)[0]
    test_dates['rp'] = test_dates['Real'].pct_change()
    test_dates['pp'] = test_dates['Pred'].pct_change()
    test_dates['s'] = test_dates['rp'] * test_dates['pp']
    test_dates['res'] = test_dates['s'] > 0
    qos = test_dates['res'].mean()
    print(qos)
    test_dates.to_csv('result.csv')
    if qos > 0.55:
        print("ALERT!! Not bad!")
        dump(Y_sc, open('Y_scaler_55.pkl', 'wb'))
        dump(sc, open('dsc_55.pkl', 'wb'))
        test_dates.to_csv('result_55.csv')
    if qos > 0.6:
        print("AAALEEERT!! HOORAAA!")
        dump(Y_sc, open('Y_scaler_06.pkl', 'wb'))
        dump(sc, open('dsc_06.pkl', 'wb'))
        test_dates.to_csv('result_06.csv')
        break
