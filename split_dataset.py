import numpy as np
import random
from print_utils import Printer

def train_test_split(X, Y, split):
    #randomly assigning split rows to training set and rest to test set
    indices = np.array(range(len(X)))
    train_size = round(split * len(X))
    random.shuffle(indices)
    train_indices = indices[0:train_size]
    test_indices = indices[train_size:len(X)]
    X_train = X[train_indices, :]
    X_test = X[test_indices, :]
    y_train = Y[train_indices, :]
    y_test = Y[test_indices, :]
    print_split_dataset(X_train,y_train, X_test, y_test)
    return X_train,y_train, X_test, y_test

def print_split_dataset(X_train,y_train, X_test, y_test):
    Printer.green("TRAINING SET")
    Printer.yellow("X_train.shape: ", X_train.shape)
    Printer.yellow("Y_train.shape: ", y_train.shape)
    Printer.green("TESTING SET")
    Printer.yellow("X_test.shape: ", X_test.shape)
    Printer.yellow("Y_test.shape: ", y_test.shape)
