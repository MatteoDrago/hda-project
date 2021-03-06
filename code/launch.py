# This script contains functions that can be used to launch preprocessing and training and testing
# of the three different types of classifications described in the paper.

import preprocessing
import models
import utils
import os
import numpy as np
from sklearn.metrics import classification_report, f1_score
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical

# Methods for preprocessing, model compilation and training

def oneshot_classification(subject, task, model_name, data_folder, window_size=15, stride=5, epochs=15, batch_size=32, balcance_classes=False, GPU=False, print_info=False):

    # preprocessing
    if task == "A":
        label = 0
    elif task == "B":
        label = 6
    else:
        print("Error: invalid task.")
    
    if subject == 23:
        X_train, Y_train, X_test, Y_test, n_features, n_classes = preprocessing.loadDataMultiple(label=label,
                                                                                                 folder=data_folder,
                                                                                                 window_size=window_size,
                                                                                                 stride=stride,
                                                                                                 make_binary=False,
                                                                                                 null_class=True,
                                                                                                 print_info=print_info)
    else:
        X_train, Y_train, X_test, Y_test, n_features, n_classes = preprocessing.loadData(subject=subject,
                                                                                         label=label,
                                                                                         folder=data_folder,
                                                                                         window_size=window_size,
                                                                                         stride=stride,
                                                                                         make_binary=False,
                                                                                         null_class=True,
                                                                                         print_info=print_info)
    # model
    if model_name == "Convolutional":
        model = models.Convolutional((window_size, n_features), n_classes, print_info=print_info)
    elif model_name == "Convolutional1DRecurrent":
        model = models.Convolutional1DRecurrent((window_size, n_features), n_classes, GPU=GPU, print_info=print_info)
    elif model_name == "Convolutional2DRecurrent":
        model = models.Convolutional2DRecurrent((window_size, n_features, 1), n_classes, GPU=GPU, print_info=print_info)
        # reshaping for 2D convolutional model
        X_train = X_train.reshape(X_train.shape[0], window_size, n_features, 1)
        X_test = X_test.reshape(X_test.shape[0], window_size, n_features, 1)
    elif model_name == "ConvolutionalDeepRecurrent":
        model = models.ConvolutionalDeepRecurrent((window_size, n_features), n_classes, GPU=GPU, print_info=print_info)
    else:
        print("Model not found.")
    model.compile(optimizer = Adam(lr=0.001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    save_model_name = task + "_" + model_name + "_OS_" + str(subject)
    filepath = './data/models/'+save_model_name+'.hdf5'
    print("Model:", save_model_name, "\nLocation:", filepath, "\n")

    # training
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
    lr_reducer = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
    model.fit(x = X_train, 
              y = to_categorical(Y_train), 
              epochs = epochs, 
              batch_size = batch_size,
              verbose = 1,
              validation_data=(X_test, to_categorical(Y_test)),
              callbacks=[checkpointer, lr_reducer])

    return model, X_test, Y_test, filepath, save_model_name

def cascade_detection(subject, task, model_name, data_folder, window_size=15, stride=5, epochs=15, batch_size=32, balcance_classes=False, GPU=False, print_info=False):

    # preprocessing
    if task == "A":
        label = 0
    elif task == "B":
        label = 6
    else:
        print("Error: invalid task.")
    if subject == 23:
        X_train, Y_train, X_test, Y_test, n_features, n_classes = preprocessing.loadDataMultiple(label=label,
                                                                                                 folder=data_folder,
                                                                                                 window_size=window_size,
                                                                                                 stride=stride,
                                                                                                 make_binary=True,
                                                                                                 null_class=True,
                                                                                                 print_info=print_info)
    else:
        X_train, Y_train, X_test, Y_test, n_features, n_classes = preprocessing.loadData(subject=subject,
                                                                                         label=label,
                                                                                         folder=data_folder,
                                                                                         window_size=window_size,
                                                                                         stride=stride,
                                                                                         make_binary=True,
                                                                                         null_class=True,
                                                                                         print_info=print_info)

    # model
    if model_name == "Convolutional":
        model = models.Convolutional((window_size, n_features), n_classes, print_info=print_info)
    elif model_name == "Convolutional1DRecurrent":
        model = models.Convolutional1DRecurrent((window_size, n_features), n_classes, GPU=GPU, print_info=print_info)
    elif model_name == "Convolutional2DRecurrent":
        model = models.Convolutional2DRecurrent((window_size, n_features, 1), n_classes, GPU=GPU, print_info=print_info)
        # reshaping for 2D convolutional model
        X_train = X_train.reshape(X_train.shape[0], window_size, n_features, 1)
        X_test = X_test.reshape(X_test.shape[0], window_size, n_features, 1)
    elif model_name == "ConvolutionalDeepRecurrent":
        model = models.ConvolutionalDeepRecurrent((window_size, n_features), n_classes, GPU=GPU, print_info=print_info)
    else:
        print("Model not found.")
    model.compile(optimizer = Adam(lr=0.001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    save_model_name = task + "_" + model_name + "_TSD_" + str(subject)
    filepath = './data/models/'+save_model_name+'.hdf5'
    print("Model:", save_model_name, "\nLocation:", filepath, "\n")

    # training
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
    lr_reducer = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
    model.fit(x = X_train, 
              y = to_categorical(Y_train), 
              epochs = epochs, 
              batch_size = batch_size,
              verbose = 1,
              validation_data=(X_test, to_categorical(Y_test)),
              callbacks=[checkpointer, lr_reducer])
    
    return model, X_test, Y_test, filepath, save_model_name

def cascade_classification(subject, task, model_name, data_folder, window_size=15, stride=5, epochs=15, batch_size=32, balcance_classes=False, GPU=False, print_info=False):

    # preprocessing
    if task == "A":
        label = 0
    elif task == "B":
        label = 6
    else:
        print("Error: invalid task.")
    if subject == 23:
        X_train, Y_train, X_test, Y_test, n_features, n_classes = preprocessing.loadDataMultiple(label=label,
                                                                                                 folder=data_folder,
                                                                                                 window_size=window_size,
                                                                                                 stride=stride,
                                                                                                 make_binary=False,
                                                                                                 null_class=False,
                                                                                                 print_info=print_info)
    else:
        X_train, Y_train, X_test, Y_test, n_features, n_classes = preprocessing.loadData(subject=subject,
                                                                                         label=label,
                                                                                         folder=data_folder,
                                                                                         window_size=window_size,
                                                                                         stride=stride,
                                                                                         make_binary=False,
                                                                                         null_class=False,
                                                                                         print_info=print_info)

    # model
    if model_name == "Convolutional":
        model = models.Convolutional((window_size, n_features), n_classes, print_info=print_info)
    elif model_name == "Convolutional1DRecurrent":
        model = models.Convolutional1DRecurrent((window_size, n_features), n_classes, GPU=GPU, print_info=print_info)
    elif model_name == "Convolutional2DRecurrent":
        model = models.Convolutional2DRecurrent((window_size, n_features, 1), n_classes, GPU=GPU, print_info=print_info)
        # reshaping for 2D convolutional model
        X_train = X_train.reshape(X_train.shape[0], window_size, n_features, 1)
        X_test = X_test.reshape(X_test.shape[0], window_size, n_features, 1)
    elif model_name == "ConvolutionalDeepRecurrent":
        model = models.ConvolutionalDeepRecurrent((window_size, n_features), n_classes, GPU=GPU, print_info=print_info)
    else:
        print("Model not found.")
    model.compile(optimizer = Adam(lr=0.001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    save_model_name = task + "_" + model_name + "_TSC_" + str(subject)
    filepath = './data/models/'+save_model_name+'.hdf5'
    print("Model:", save_model_name, "\nLocation:", filepath, "\n")

    # training
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
    lr_reducer = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
    model.fit(x = X_train, 
              y = to_categorical(Y_train), 
              epochs = epochs, 
              batch_size = batch_size,
              verbose = 1,
              validation_data=(X_test, to_categorical(Y_test)),
              callbacks=[checkpointer, lr_reducer])

    
    return model, X_test, Y_test, filepath, save_model_name

# Method for testing and evaluating results

def evaluation(model, X_test, Y_test, filepath, save_model_name):

    # last model
    Y_pred = model.predict_classes(X_test)
    score = f1_score(Y_test, Y_pred, average='weighted')

    # best model
    model_best = load_model(filepath)
    Y_pred_best = model_best.predict_classes(X_test)
    score_best = f1_score(Y_test, Y_pred_best, average='weighted')

    # keep highest f1-score
    if score_best > score:
        print("\nResults for best "+ save_model_name + ":\n", classification_report(Y_test, Y_pred_best))
        return Y_pred_best, score_best
    else:
        print("\nResults for last "+ save_model_name + ":\n", classification_report(Y_test, Y_pred))
        return Y_pred, score
