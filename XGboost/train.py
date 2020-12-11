from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import numpy as np


PROCESSED_DATA_BASE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/../hari_data_processed/'

def load_data():
        print("Starting to loading train data..")
        #X is already transfored with TfidfVectorizer
        X_train = pickle.load(open(PROCESSED_DATA_BASE_PATH + 'train/X', 'rb'))
        X_train = X_train.todense()
        Y_train = pickle.load(open(PROCESSED_DATA_BASE_PATH + 'train/Y', 'rb'))
        print("Finished loading train data..")

        print("Starting to loading test data..")
        #X is already transfored with TfidfVectorizer
        X_test = pickle.load(open(PROCESSED_DATA_BASE_PATH + 'test/X', 'rb'))
        X_test = X_test.todense()
        Y_test = pickle.load(open(PROCESSED_DATA_BASE_PATH + 'test/Y', 'rb'))
        print("Finished loading test data..")

        return X_train, np.array(Y_train), X_test, np.array(Y_test)

def train():
    X_train, Y_train, X_test, Y_test = load_data()

    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}, X_test: {X_test.shape}, Y_test: {Y_test.shape}")

    xgb_params = {
        'random_state' :23,
        'seed': 2,
        'colsample_bytree' : 0.6,
        'subsample' : 0.7,
        'tree_method' :'hist',
        'max_depth' :10, 
        'n_estimators' : 2,
        'objective' : 'binary:logistic',
        'use_label_encoder' : False,
    }

    label_encoder = LabelEncoder().fit(Y_train)
    label_encoder_y_train = label_encoder.transform(Y_train)
    label_encoder_y_test =  label_encoder.transform(Y_test)
    # fit model no training data
    xgb_model = XGBClassifier(**xgb_params)
    bst = xgb_model.fit(X_train, label_encoder_y_train.ravel())

    bst.save_model('xgb.model')



    y_pred = xgb_model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    print(f"Accuracy: {accuracy_score(label_encoder_y_test, y_pred)}")

    print(f"F1-Score: {f1_score(label_encoder_y_test, y_pred, average=None)[0]}")

if __name__ == "__main__":
    train()
