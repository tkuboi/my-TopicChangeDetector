import sys
import traceback
import numpy as np

from functools import partial
from utils import *

from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D, Embedding, RepeatVector, Lambda, Dot, Multiply, Concatenate, Permute, MaxPooling1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, Flatten, ThresholdedReLU
from keras.optimizers import Adam

UTTERANCE_SIZE = 200
LABELS = ['False', 'True']

class TopicChangeDetector:
    """A class which incorporates a Keras model and methods to work on the model.
    The class also includes a function to create labeled data, evaluating the model
    , and adding punctuations to text.
    The class is designed to work on English text.
    """
 
    def __init__(self, words_to_index, model):
        """Constructor
        Args:
            words_to_index: a dictionary mapping words to index
            model: a Keras model
        """
        self.size = UTTERANCE_SIZE
        self.labels = LABELS
        self.words_to_index = words_to_index
        self.__model__ = model
        self.score = {"correct":0, "wrong":0, "TP":0, "FP":0, "TN":0, "FN":0}

    def reset_score(self):
        self.score = {"correct":0, "wrong":0, "TP":0, "FP":0, "TN":0, "FN":0}

    def compile(self, opt, **kwargs):
        """Compile a Keras model with specified options
        Args:
            opt: optimizer
            **kwargs: a dictionary for other options
        """
        self.__model__.compile(opt, **kwargs) 

    def fit(self, train_X, train_Y, batch_size=10, epochs=1):
        """ calls fit method on the Keras model with options
        Args:
            train_X: training data
            train_Y: labels for the trainig data
            batch_size: batch size
            epochs: the number of epochs
        """
        self.__model__.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs)

    def save_model(self, filename):
        """Save a Keras model to a json file
        Args:
            filename: a path name to a file to save a model in
        """
        save_model(self.__model__, filename)

    def save_weights(self, filename):
        """Save weights to a file
        Args:
            filename: a path name to a file to save weights in
        """
        self.__model__.save_weights(filename)

    def load_model(self, filename):
        """Load a Keras model from a json file
        Args:
            filename: a path name to a file to load a model from
        """
        json_file = open(filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.__model__ = model_from_json(loaded_model_json, custom_objects={'softmax2': softmax2})

    def load_weights(self, filename):
        """Load weights from a file
        Args:
            filename: a path name to a file to load weights from
        """
        self.__model__.load_weights(filename)

    def evaluate(self, test_X, test_Y, batch_size=10):
        """Evaluate the model by calling the Keras model's evaluate method
        Args:
            test_X:
            test_Y:
            batch_size:
            epochs:
        Returns:
           Scalar test loss
        """ 
        return self.__model__.evaluate(
            test_X, test_Y, batch_size=batch_size)

    def predict(self, x):
        """Make a prediction by calling the Keras model's predict method
        Args:
            x: a list of index to word
        Returns:
            an array of confidence value per label per word index 
        """
        return self.__model__.predict([np.array([x])])

    def create_training_data(self, utterances, repeat_size=1):
        """Create training data by labeling
        Args:
            utterances: list of utterance(uid, did, start, end, text)
        Returns:
            (x, y): x - list of utterances modified to fit to the model
                    y - list of labels ordered in the same order as x
        """
        m = len(utterances) * 3 
        x =  np.zeros((m, self.size))
        y = np.zeros((m, len(self.labels))) 
        dids = []
        cur_did = None
        label = 0
        count = 0
        rec_count = 0
        for uid,did,start,end,text in utterances:
            if did != cur_did or (label == 1 and count < repeat_size):
                label = 1
                count += 1
                cur_did = did
            else:
                label = 0
                count = 0

            words = self.convert_to_index(self.fit_data_size(text))
            x[rec_count:] = np.asarray(words)
            for i in range(0,len(words)):
                y[rec_count+i][label] = 1
            rec_count += len(words)
            """if label == 1:
                for i in range(0,3):
                    x[rec_count:] = np.asarray(words)
                    y[rec_count:] = np.asarray([[label]] * len(words))
                    rec_count += len(words)
            """

        return x[:rec_count],y[:rec_count]

    def convert_to_index(self, words):
        results = []
        for line in words:
            indexes = []
            for word in line:
                indexes.append(self.lookup_index(word))
            results.append(indexes)
        return results

    def lookup_index(self, word):
        """A private helper function to
          convert word to index and store it in 2D list
        Args:
            i: index for i th example
            j: index for j th word
            X: 2D list of index to word
            word: the j th word
        """
        word = word.lower().strip()
        idx = self.words_to_index.get(word)
        if idx is None:
            idx = self.words_to_index.get('[UNKNOWN]')
        return idx

    def fit_data_size(self, text):
        rest = text.split()
        results = [] 
        count = 0 
        while len(rest) > self.size:
            results.append(rest[:self.size])
            rest = rest[self.size:]
            count += 1
        rest.extend([""] * (self.size - len(rest)))
        results.append(rest)
        return results

    def create_live_data(self, utterances):
        """Create live data to be fed to the model 
        Args:
            utterances: list of utterances
        Returns:
            x: list of utterances modified to fit to the model
            y: zero filled matrix
        """
        return create_training_data(self, utterances, 0)

    def check_result(self, prediction, label):
        if prediction == 1:
            if prediction == label:
                self.score['TP'] += 1
            else:
                self.score['FP'] += 1
        else:
            if prediction == label:
                self.score['TN'] += 1
            else:
                self.score['FN'] += 1

    def test_model(self, X, Y):
        for x,y in zip(X, Y):
            result = self.predict(x)
            y_c = np.argmax(y, axis=-1)
            y_hat = np.argmax(result, axis=-1)
            self.check_result(y_hat[0], y_c)
            print(result, y_hat, y_c)
     
 
