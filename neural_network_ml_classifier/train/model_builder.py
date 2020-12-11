import math
import tensorflow as tf


def build(feature_count):
    """Builds the neural network model using tensorflow

        Arguments:
            feature_count: total # feature intend to provide to the model

        :returns:
            tf.keras.models.Sequential: Fully formed neural network model with input, hidden and output layers.
        """
    print("Preparing model with feature count: ", feature_count)
    nn = tf.keras.models.Sequential()

    nn.add(tf.keras.Input(shape=(feature_count,)))
    nn.add(tf.keras.layers.Dense(math.ceil(feature_count / 1000), activation='relu'))
    nn.add(tf.keras.layers.Dense(math.ceil(feature_count / 1000 / 4), activation='relu'))
    nn.add(tf.keras.layers.Dense(math.ceil(feature_count / 1000 / 8), activation='relu'))
    nn.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    print("finished preparing model..")
    return nn
