"""This script is for training and evaluating a model."""

import sys
import os
import traceback
import numpy as np
from functools import partial
from utils import *
from topic_change_detector import TopicChangeDetector
from bidirectional_gru_with_dense import BidirectionalGruWithDense
from bidirectional_gru_with_conv import BidirectionalGruWithConv
from bidirectional_gru_with_conv_v2 import BidirectionalGruWithConvV2
from bidirectional_gru_with_conv_v3 import BidirectionalGruWithConvV3
from conv_with_dense import ConvWithDense
from multi_kernel_conv_with_dense import MultiKernelConvWithDense

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D, Embedding, RepeatVector, Lambda, Dot, Multiply, Concatenate, Permute
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, Flatten, ThresholdedReLU
from keras.optimizers import Adam

EMBEDDING_FILE = 'data/glove.6B.50d.txt'
MODEL_FILE = 'data/model3.json'
WEIGHTS_FILE = 'data/model3.h5'
TEXT_FILE = 'data/training_text_1000.txt'
BATCH = 10 
EPOCH = 1 
DEV_SIZE = 100 

def load_text_data(textfile):
    """Read a text file containing lines of text.
    Args:
        textfile: string representing a path name to a file
    Returns:
        list of words
    """
    utterances = []
    with open(textfile, 'r') as lines:
        for line in lines:
            utterances.append(tuple(line.split("\t")))
    return utterances 

def check_results(detector, results, labels):
    for result, label in zip(results, labels):
        detector.check_result(result, label)

def main():
    """Train a model using lines of text contained in a file
    and evaluates the model.
    """

    #read golve vecs
    words, word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(EMBEDDING_FILE)
    print(index_to_word[400001])
    #create word embedding matrix
    embedding_matrix = create_emb_matrix(word_to_index, word_to_vec_map)
    print('shape of embedding_matrix:', embedding_matrix.shape)

    #load trainig text from a file
    utterances = load_text_data(TEXT_FILE)
    print(utterances[0])

    #create an instance of Punctutor and create training data
    detector = TopicChangeDetector(word_to_index, None)
    X, Y, Z = detector.create_training_data(utterances, 1, 10, True)

    #if a model already exists, load the model
    if os.path.isfile(MODEL_FILE):
        detector.load_model(MODEL_FILE)
    else: 
        """model = BidirectionalGruWithDense.create_model(
            input_shape=(X.shape[1], ), embedding_matrix=embedding_matrix,
            vocab_len=len(word_to_index), n_d1=128, n_d2=128, n_c=len(detector.labels))
        """
        model = MultiKernelConvWithDense.create_model(
            input_shape=(X.shape[1], ), embedding_matrix=embedding_matrix,
            vocab_len=len(word_to_index), n_d1=128, n_d2=64, n_c=len(detector.labels))
        print(model.summary())
        detector.__model__ = model

    #if the model has been already trained, use the pre-trained weights
    if os.path.isfile(WEIGHTS_FILE): 
        detector.load_weights(WEIGHTS_FILE)

    #shuffle the training data
    shuffle(X,Y,Z)
 
    denom_Y = Y.swapaxes(0,1).sum((0,1))
    print ('Summary of Y:', denom_Y)

    print('shape of X:', X.shape)
    print(X[0:1]) 
    print('shape of Y:', Y.shape)
    #print(Y[0:10])

    #define optimizer and compile the model
    opt = Adam(lr=0.007, beta_1=0.9, beta_2=0.999, decay=0.01)
    detector.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

    #split the training data into training set, test set, and dev set
    t_size = int(X.shape[0] * 0.9)
    train_X, train_Y, train_Z = X[:t_size], Y[:t_size], Z[:t_size]
    test_X, test_Y, test_Z = X[t_size:-DEV_SIZE], Y[t_size:-DEV_SIZE], Y[t_size:-DEV_SIZE]
    dev_X, dev_Y, dev_Z = X[-DEV_SIZE:], Y[-DEV_SIZE:], Z[-DEV_SIZE:]

    print (train_Y.swapaxes(0,1).sum((0,1)))
    print (test_Y.swapaxes(0,1).sum((0,1)))

    #train the model
    detector.fit([train_X], train_Y, batch_size = BATCH,
               epochs=EPOCH)
    detector.save_model(MODEL_FILE)
    detector.save_weights(WEIGHTS_FILE)

    print('shape of test_X:', test_X.shape)
    print('shape of test_Y:', test_Y.shape)

    evaluation = detector.evaluate([test_X], test_Y)
    print(evaluation)

    print('shape of dev_X:', dev_X.shape)
    print('shape of dev_Y:', dev_Y.shape)
    detector.test_model(dev_X, dev_Y, dev_Z)
    print(detector.print_score())

if __name__ == "__main__":
    main()
