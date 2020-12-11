import math

import tensorflow as tf

def build(input_batch_size, feature_count):
    print("Preparing model with feature count: ", feature_count)
    nn = tf.keras.models.Sequential()

    nn.add(tf.keras.Input(shape=(feature_count,)))
    nn.add(tf.keras.layers.Dense(math.ceil(feature_count/1000), activation='relu'))
    nn.add(tf.keras.layers.Dense(math.ceil(feature_count/1000/4), activation='relu'))
    nn.add(tf.keras.layers.Dense(math.ceil(feature_count/1000/8), activation='relu'))
    nn.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    print("finished preparing model..")
    return nn
