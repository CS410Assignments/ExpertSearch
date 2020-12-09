import math
import os

import tensorflow as tf
import pathlib
import tempfile
import shutil

from tensorflow import float32, float64
from joblib import dump, load
import os

name = 'stock_pred_project'
logdir = "/home/dl_models/logs/" + name + "/logs"
shutil.rmtree(logdir, ignore_errors=True)

checkpoint_path = "/home/dl_models/logs/" + name + "/"


checkpoint_dir = os.path.dirname(checkpoint_path)


def compile_fit(model, train_set, val_set, train_set_length, val_set_length, batch_size):

    steps_per_epoch = math.ceil(train_set_length/batch_size)
    validation_steps = math.ceil(val_set_length/batch_size)
    print("starting the training. Steps/epoch: ", steps_per_epoch, " and validation steps: ", validation_steps)
    # lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    #     0.001,
    #     decay_steps=steps_per_epoch * 1000,
    #     decay_rate=1,
    #     staircase=False)

    # Last good lr = 0.00005
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.00008)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

    # tf.keras.losses.MeanAbsoluteError(
    #     reduction=tf.keras.losses.Reduction.AUTO)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'mean_absolute_error'
            # tf.keras.metrics.MeanSquaredError()
        ])

    model_history = model.fit(train_set,
                              epochs=20,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=val_set,
                              validation_steps=validation_steps,
                              callbacks=get_callbacks()
                              )
    return model, model_history


def get_callbacks():
    check_point = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1)

    return [
        tf.keras.callbacks.EarlyStopping(patience=100),
        # tf.keras.callbacks.TensorBoard(logdir/name),
        # check_point
    ]

def saveModels(model, vectorizer, target_dir_root):
    print("Saving the models to: ", target_dir_root)

    model_dir = target_dir_root + "model/"
    vec_dir = target_dir_root + "vectorizer/"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(vec_dir):
        os.makedirs(vec_dir)

    model.save(model_dir)
    dump(vectorizer, vec_dir + "tfidf")

    print("Finished saving the models.....")