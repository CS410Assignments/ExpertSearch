from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import pandas as pd
from pandas import DataFrame
from sklearn.externals import joblib

from utils.data_utils import build_feature, cleanup_expand
from model.lstm.model_builder import build, plot_train_history, predict_next_day

from model.lstm.model_executor import compile_fit
from sklearn.model_selection import train_test_split
import numpy as np

train_data_precentage = 0.7
MINI_BATCH_SIZE = 64
history_size = 30
predict_size = 7

target_column_name = 'High'

x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

input_data: DataFrame = pd.read_csv('/home/dl_models/Data/stocks/BE/BE_05_26_2020.csv')

input_data = input_data[70:]

print("Input data shape: ", input_data.shape)
print(input_data.head())

input_data = cleanup_expand(input_data)

target_column_index = list(input_data.columns.values).index(target_column_name)
input_data = input_data.__array__()

# scale down input data
x_scaler.fit(input_data)
scaled_input_data = x_scaler.transform(input_data)

target_column = np.array(input_data[:, target_column_index]).reshape(len(scaled_input_data), 1)

y_scaler.fit(target_column)
scaled_target_column = y_scaler.transform(target_column)

X, Y = build_feature(scaled_input_data, scaled_target_column, history_size, predict_size)

print("X length", len(X), " and Y length", len(Y))

################### Data is Ready ######################

# train_set_X, train_set_Y, val_set_X, val_set_Y = shuffle_split_train_val_sets(X, Y, train_data_precentage)
train_set_X, val_set_X, train_set_Y, val_set_Y = train_test_split(X, Y, test_size=1-train_data_precentage, shuffle=True)

train_set_X, val_set_X, train_set_Y, val_set_Y = np.array(train_set_X), np.array(val_set_X), np.array(train_set_Y), np.array(val_set_Y )
train_set = tf.data.Dataset.from_tensor_slices((train_set_X, train_set_Y))
train_set = train_set.shuffle(10000).batch(MINI_BATCH_SIZE).repeat()

val_set = tf.data.Dataset.from_tensor_slices((val_set_X, val_set_Y))
val_set = val_set.batch(MINI_BATCH_SIZE).repeat()

input_shape_for_model = train_set_X.shape[-2:]
lstm_model = build(input_shape_for_model, predict_size)
compile_fit(lstm_model, train_set, val_set, train_set_X.shape[0], val_set_X.shape[0], MINI_BATCH_SIZE)

############# Model trained #####################

plot_train_history(lstm_model.history, 'Train & Val loss')

val_predicted = predict_next_day(scaled_input_data, history_size, lstm_model, y_scaler)

# Save model
target_dir = "/home/dl_models/logs/stock_project/"+target_column_name
lstm_model.save(target_dir)
joblib.dump(x_scaler, target_dir+"/X_scaler")
joblib.dump(y_scaler, target_dir+"/Y_scaler")