import itertools

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
from sklearn.preprocessing import MinMaxScaler

import misc

close_col_idx = 11


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


filename = "xau_lstm.csv"
df = pd.read_csv(filename)
print(df.info())

df['Date'] = pd.to_datetime(df['Date'])
df.set_axis(df['Date'], inplace=True)
# df.drop(columns=['Open', 'High', 'Low', 'Currency'], inplace=True)

# data_prices = df.drop(['Open', 'Low', 'High'], axis=1)
df['EMA_9'] = df['Close'].ewm(9).mean().shift()
df = misc.applyFeatures(df)
df = df.iloc[51:]  # Because of moving averages and MACD line and bollinger
df = df[:-1]  # Because of shifting close price
df = df.dropna()
data_prices = df.drop('Date', axis=1)

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
data_prices_scaled = sc.fit_transform(data_prices)

# Number of days you want to predict into the future
# Number of past days you want to use to predict the future

X_train, X_test, X_val, y_train, y_test, y_val = create_data(data_prices_scaled, n_future=1, n_past=25,
                                                             train_test_split_percentage=0.8,
                                                             validation_split_percentage=0)
# ------------------LSTM-----------------------
regressor = Sequential()
regressor.add(LSTM(units=16, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=16, return_sequences=False))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1, activation='linear'))
regressor.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

print(regressor.summary())

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
# mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# fit model
# history = regressor.fit(X_train, y_train, validation_split=0.3, epochs=40, batch_size=64, callbacks=[es])

# print(history.history.keys())
#
# fig = plt.figure(figsize=(20, 7))
# fig.add_subplot(121)
#
# # Accuracy
# plt.plot(history.epoch, history.history['root_mean_squared_error'], label="rmse")
# plt.plot(history.epoch, history.history['val_root_mean_squared_error'], label="val_rmse")

# plt.title("RMSE", fontsize=18)
# plt.xlabel("Epochs", fontsize=15)
# plt.ylabel("RMSE", fontsize=15)
# plt.grid(alpha=0.3)
# plt.legend()
#
# # Adding Subplot 1 (For Loss)
# fig.add_subplot(122)

# plt.plot(history.epoch, history.history['loss'], label="loss")
# plt.plot(history.epoch, history.history['val_loss'], label="val_loss")
#
# plt.title("Loss", fontsize=18)
# plt.xlabel("Epochs", fontsize=15)
# plt.ylabel("Loss", fontsize=15)
# plt.grid(alpha=0.3)
# plt.legend()
#
# plt.show()
#
# results = regressor.evaluate(X_test, y_test)
# print("test loss, test acc:", np.round(results, 4))

config = [[False], [False], [False], [16, 32, 64], [8, 16, 32, 64], [0.1, 0.11, 0.12, 0.13,
                                                                     0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.3]]

# list of lists -->
# [[first_additional_layer], [second_additional_layer], [third_additional_layer], [n_neurons], [n_batch_size], [dropout]]
hist = LSTM_HyperParameter_Tuning(config, X_train, y_train, X_test, y_test)  # change x_train shape

hist = pd.DataFrame(hist)
hist = hist.sort_values(by=[7], ascending=True)
hist.to_csv('hist_3_add.csv')
print(hist)
