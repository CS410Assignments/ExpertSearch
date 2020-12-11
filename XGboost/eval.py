from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import pickle

def load_data():

    print("Starting to loading test data..")
    #X is already transfored with TfidfVectorizer
    X_test = pickle.load(open(PROCESSED_DATA_BASE_PATH + 'test/X', 'rb'))
    X_test = X_test.todense()
    Y_test = pickle.load(open(PROCESSED_DATA_BASE_PATH + 'test/Y', 'rb'))
    print("Finished loading test data..")

    return X_train, Y_train, X_test, Y_test