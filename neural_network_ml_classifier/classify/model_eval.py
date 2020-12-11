import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential

data_base_path = '/hari_data_processed/untouch/test/'
model_base_path = '/Users/hbojja/PycharmProjects/trained_models/expert_search/'


print("Starting to loading models..")
model:Sequential = tf.keras.models.load_model(model_base_path + 'model')

# vectorizer:TfidfVectorizer = pickle.load(open(model_base_path + 'vectorizer/tfidf', 'rb'))

print("Starting to loading test data..")
#X is already transfored with TfidfVectorizer
X_test = pickle.load(open(data_base_path + 'X', 'rb'))
X_test = X_test.todense()
Y_test = pickle.load(open(data_base_path + 'Y', 'rb'))
print("Finished loading test data..")

print("Starting predictions in test data..")
predictions = model.predict(X_test)
predictions_classes = np.where(predictions > 0.5, 1, 0)

print("Finished all predictions..")

match_count = 0
for idx in range(len(predictions)):
    actual = Y_test[idx][0]
    pred_val = predictions_classes[idx][0]
    if actual == pred_val:
        match_count += 1
    else:
        print("Actual: ", actual, " and pred class: ", pred_val, " pred val: ", predictions[idx][0])

print("Accurate prediction count", match_count)
print("Accurate prediction percentage", match_count/len(Y_test)*100)
