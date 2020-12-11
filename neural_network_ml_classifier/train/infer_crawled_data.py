import math

import tensorflow as tf
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
import numpy as np
from neural_network_ml_classifier.PreProcess import PreProcessor

crawled_data_path = '../../Crawl-n-Extract/Merge/UIUC.txt'
model_base_path = '../fully_trained/'

print("Loading models models..")
model:Sequential = tf.keras.models.load_model(model_base_path + 'NN_V1/model')
vectorizer:TfidfVectorizer = pickle.load(open(model_base_path + 'vectorizer/vectorizer_object', 'rb'))
# vectorizer:TfidfVectorizer = pickle.load(open('/Users/hbojja/uiuc/CS410-TIS/ExpertSearch/hari_data_processed/untouch/vectorizer_object', 'rb'))

print("Loading models completed..")

print("Loading crawled data..")

crawled_data = open(crawled_data_path, 'r').readlines()

print("Loading crawled data completed..")

pp = PreProcessor()
processed_lines = []
counter = 0
crawled_data_len = len(crawled_data)
for line in crawled_data:
    line_split = line.split('#####')
    if(len(line_split) < 2):
        continue

    processed_line = pp.intersectStopWordsAndStem(line_split[1].strip())
    processed_lines.append(processed_line)
    counter += 1
    if counter % 100 == 0:
        print("Pre-processing data. Completed: ", counter, "/", crawled_data_len)

print("Pre-processing data. Completed: ", crawled_data_len, "/", crawled_data_len)

print("Working on classifying the faculty pages using pre-trained neural network model..")
processed_line_vec = vectorizer.transform(processed_lines)
predicted_values = model.predict(processed_line_vec)
predicted_values_labeled = np.where(predicted_values > 0.7, 1, 0)
faculty_count = 0
for idx in range(len(predicted_values_labeled)):
    if predicted_values_labeled[idx][0] == 1:
        faculty_count += 1
        print(str(predicted_values_labeled[idx][0]) + "     " + crawled_data[idx])

