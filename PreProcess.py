from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer;
import os

# One time step
# python -m nltk.downloader stopwords
# nltk.download('stopwords')
#  python -m nltk.downloader punkt
class PreProcessor(object):
    stop_words = None
    private_stop_word_list = {'.', ',', ':', ';'}
    stemmer = None

    def __init__(self):
        # init stopwords
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(self.private_stop_word_list)

        # init Stemmer
        self.stemmer = PorterStemmer()

    # The stop word removal and stemming is performed in the same function to keep the cost to 'n'.
    # If written separately, cost could go up to '3n'.
    def intersectStopWordsAndStem(self, line):
        pre_tokens = word_tokenize(line)
        final_string = ''
        for w in pre_tokens:
            if w in self.stop_words:
                continue
            final_string += ' ' + self.stemmer.stem(w)

        return final_string

    def process(self, source_dir, dest_dir):
        for file in os.listdir(source_dir):
            dest_file = open(dest_dir + "/" + file, mode='w')
            for line in open(source_dir + "/" + file):
                dest_file.write(self.intersectStopWordsAndStem(line) + "\n")
            dest_file.close()


preProcessor = PreProcessor()
preProcessor.process("./raw_data", "./pre_processed_data")
# preProcessor.process("Hello, This is a sample sentence with full of stop words. "
#                      "Stems the words and prints the results. No ordered manner.")
