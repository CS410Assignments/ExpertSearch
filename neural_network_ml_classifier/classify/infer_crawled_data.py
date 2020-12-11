import tensorflow as tf
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
import numpy as np
from neural_network_ml_classifier.data_processor.PreProcess import PreProcessor


"""
    This script reads the crawled data from path defined variable "crawled_data_path", trained model from the path defined in "model_base_path".
    Uses the trained model to classify faculty pages.
    
    :input locations
        crawled_data_path: File path where crawled data is stored.
        model_base_path: Path to tensorflow ML model fully trained and saved.
"""
crawled_data_path = '../../Crawl-n-Extract/Merge/UIUC.txt'
model_base_path = '../../fully_trained_model/'


print("Loading trained model from: ", model_base_path + 'neural_network_model_v1/model')
model:Sequential = tf.keras.models.load_model(model_base_path + 'neural_network_model_v1/model')

print("Loading vectorized from: ", model_base_path + 'vectorizer/vectorizer_object')
vectorizer:TfidfVectorizer = pickle.load(open(model_base_path + 'vectorizer/vectorizer_object', 'rb'))

print("Loading trained models completed..")

print("Loading crawled data..")

crawled_data = open(crawled_data_path, 'r').readlines()

print("Loading crawled data completed. # docs read: ", len(crawled_data))


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
predicted_values_labeled = np.where(predicted_values > 0.5, 1, 0)

faculty_count = 0
print("Printing the pages classified as faculty pages..")
for idx in range(len(predicted_values_labeled)):
    if predicted_values_labeled[idx][0] == 1:
        faculty_count += 1
        print(crawled_data[idx])

