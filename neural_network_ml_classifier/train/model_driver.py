import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from neural_network_ml_classifier.train.model_builder import build
import pickle
from neural_network_ml_classifier.train.model_executor import compile_fit, saveModels

data_base_dir = '/hari_data_processed/untouch/'
train_data_precentage = .8
MINI_BATCH_SIZE = 64

print("starting Input data preparation finished...")
X = pickle.load(open(data_base_dir + '/train/X', 'rb'))
#convert sparse matrix to dense
X = X.todense()

Y = pickle.load(open(data_base_dir + '/train/Y', 'rb'))

vectorizer = pickle.load(open(data_base_dir + '/vectorizer_object', 'rb'))
print("Input data preparation finished...")

train_set_X, val_set_X, train_set_Y, val_set_Y = train_test_split(X, Y, test_size=1-train_data_precentage, shuffle=True)

train_set_X, val_set_X, train_set_Y, val_set_Y = np.array(train_set_X), np.array(val_set_X), np.array(train_set_Y), np.array(val_set_Y )
train_set = tf.data.Dataset.from_tensor_slices((train_set_X, train_set_Y))
train_set = train_set.shuffle(10000).batch(MINI_BATCH_SIZE).repeat()

val_set = tf.data.Dataset.from_tensor_slices((val_set_X, val_set_Y))
val_set = val_set.batch(MINI_BATCH_SIZE).repeat()

neural_network = build(len(vectorizer.get_feature_names()))

model, model_history = compile_fit(neural_network, train_set, val_set, train_set_X.shape[0], val_set_X.shape[0], MINI_BATCH_SIZE)

saveModels(model, vectorizer, '/Users/hbojja/PycharmProjects/trained_models/expert_search/')