from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import os
import pickle
import numpy as np
import joblib

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

def main():
    X_train, Y_train, X_test, Y_test = load_data()

    # fit model no training data
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, Y_train.ravel())

    y_pred = logisticRegr.predict(X_test)
    predictions = [round(value) for value in y_pred]

    print(f"F1:score: {f1_score(Y_test, y_pred, average=None)[0]}")
    joblib.dump(logisticRegr, "logit.model")

if __name__ == "__main__":
    main()
