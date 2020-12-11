import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split

# This script reads the data from the paths defined in "pos_data", "neg_data" variables
# Shuffles the data and split in to 85% training set, 15% test set.
# pos_data: positive examples
# neg_data : negative examples
def writeXToFile(data, fileName):
    data_file = open(fileName, "w")
    for line in data:
        if not isinstance(line, str):
            continue

        data_file.write(line)

    data_file.close()


def writeToFile(data_ip, file_name):
    data_file = open(file_name, 'wb')
    pickle.dump(data_ip, data_file)
    data_file.close()


pos_data = open('../../hari_data_processed/positive.txt').readlines()
neg_data = open('../../hari_data_processed/negative.txt').readlines()

print("starting Input data preparation finished...")
all_data = pos_data + neg_data

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(all_data)

Y = np.append(np.ones(len(pos_data)), np.zeros(len(neg_data)))
Y = np.transpose(np.asmatrix(Y, dtype=np.float64))

train_set_X, test_set_X, train_set_Y, test_set_Y = train_test_split(X, Y, test_size=.15, shuffle=True)

writeToFile(train_set_X, "../../hari_data_processed/train/X")
writeToFile(train_set_Y, "../../hari_data_processed/train/Y")
writeToFile(test_set_X, "../../hari_data_processed/test/X")
writeToFile(test_set_Y, "../../hari_data_processed/test/Y")
writeToFile(vectorizer, "../../hari_data_processed/vectorizer_object")
