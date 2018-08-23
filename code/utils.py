# Auxiliary functions for HDA project on Human Activity Recognition

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix, accuracy_score

import itertools
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    from keras import regularizers
    from keras.layers import Conv1D, Conv2D, BatchNormalization, Dropout, LeakyReLU, Flatten, Activation, Dense, MaxPooling1D, MaxPooling2D
    from keras.models import Model, Sequential
    from keras.optimizers import Adam
    import keras.backend as K


def loadData(subject, folder="./",  printInfo = False):
    """ Load ADL1 to ADL5 and Drill .mat files for a subject. """
    
    filename_1 = folder + "S" + str(subject) + "-ADL1"
    filename_2 = folder + "S" + str(subject) + "-ADL2"
    filename_3 = folder + "S" + str(subject) + "-ADL3"
    filename_4 = folder + "S" + str(subject) + "-ADL4"
    filename_5 = folder + "S" + str(subject) + "-ADL5"
    filename_6 = folder + "S" + str(subject) + "-Drill"

    data1 = scipy.io.loadmat(filename_1, mdict={'features_interp':'features', 'labels_cut':'labels'})
    data2 = scipy.io.loadmat(filename_2, mdict={'features_interp':'features', 'labels_cut':'labels'})
    data3 = scipy.io.loadmat(filename_3, mdict={'features_interp':'features', 'labels_cut':'labels'})
    data4 = scipy.io.loadmat(filename_4, mdict={'features_interp':'features', 'labels_cut':'labels'})
    data5 = scipy.io.loadmat(filename_5, mdict={'features_interp':'features', 'labels_cut':'labels'})
    data6 = scipy.io.loadmat(filename_6, mdict={'features_interp':'features', 'labels_cut':'labels'})

    if (printInfo):
        print("\nSession shapes:")
        print("ADL1:  ", data1['features_interp'].shape)
        print("ADL2:  ", data2['features_interp'].shape)
        print("ADL3:  ", data3['features_interp'].shape)
        print("ADL4:  ", data4['features_interp'].shape)
        print("ADL5:  ", data5['features_interp'].shape)
        print("Drill: ", data6['features_interp'].shape)

    return (data1, data2, data3, data4, data5, data6)

def prepareData(X, Y, window_size=15, stride=15, printInfo = False, null_class = True):
    """ Prepare data in windows to be passed to the model. """

    samples, features = X.shape
    classes = Y.shape[1]
    # shape output
    windows = int(samples // stride) - int(window_size // stride) 
    X_out = np.zeros([windows, window_size, features])
    Y_out = np.zeros([windows, classes])
    # write output
    for i in range(windows):
        index = int(i * stride)
        X_out[i, :, :] = X[index:index+window_size, :].reshape((window_size,features))
        temp = Y[index:index+window_size, :]
        Y_out[i, np.argmax(np.sum(temp, axis=0))] = 1 # hard version      CHECK!

    if not(null_class):
        non_null = (Y_out[:,0] == 0) # samples 0-labeled (Y_out[:,0] is the first column of Y_out)
        X_out = X_out[non_null]
        Y_out_new = Y_out[non_null][:,1:]
        Y_out = Y_out_new

    if (printInfo):
        print("Dataset of Images have shape: ", X_out.shape,\
            "\nDataset of Labels have shape:   ", Y_out.shape,\
            "\nFraction of labels:  ", np.sum(Y_out, axis=0) / Y_out.shape[0])

    return (X_out, Y_out)

def prepareDataDFT(X, Y, window_size=15, stride=15, infoDFT='angle', printInfo = True, null_class = True):
    """ Prepare data in windows to be passed to the CNN. """

    samples, features = X.shape
    classes = Y.shape[1]
    # shape output
    windows = int(samples // stride) - int(window_size // stride) 
    X_out = np.zeros([windows, window_size, features])
    Y_out = np.zeros([windows, classes])
    # write output
    for i in range(windows):
        index = int(i * stride)
        input_train = np.fft.fft2(X[index:index+window_size, :].reshape((window_size,features)))
        if (infoDFT == 'angle'):
            X_out[i, :, :] = np.angle(X[index:index+window_size, :].reshape((window_size,features)))
        else: 
            X_out[i, :, :] = np.abs(X[index:index+window_size, :].reshape((window_size,features)))
        temp = Y[index:index+window_size, :]
        Y_out[i, np.argmax(np.sum(temp, axis=0))] = 1 # hard version      CHECK!

    if not(null_class):
        non_null = (Y_out[:,0] == 0) # samples 0-labeled (Y_out[:,0] is the first column of Y_out)
        X_out = X_out[non_null]
        Y_out_new = Y_out[non_null][:,1:]
        Y_out = Y_out_new

    if (printInfo):
        print("Features have shape: ", X_out.shape,\
            "\nLabels have shape:   ", Y_out.shape,\
            "\nFraction of labels:  ", np.sum(Y_out, axis=0) / Y_out.shape[0])

    return (X_out, Y_out)

def preprocessing(subject, folder="./", label=0, window_size=15, stride=15, make_binary=False, null_class = True, printInfo = False):
    """Load, concatenate and prepare sessions of a single subject to be passed to the model."""

    # import all sessions for a subject
    (data1, data2, data3, data4, data5, data6) = loadData(subject, folder=folder)

    # create training set and test set
    X_train = np.concatenate((data1['features_interp'],\
                              data2['features_interp'],\
                              data3['features_interp'],\
                              data6['features_interp']), axis=0)

    Y_train = np.concatenate((data1['labels_cut'][:,label],\
                              data2['labels_cut'][:,label],\
                              data3['labels_cut'][:,label],\
                              data6['labels_cut'][:,label]), axis=0)

    X_test = np.concatenate((data4['features_interp'],\
                             data5['features_interp']), axis=0)

    Y_test = np.concatenate((data4['labels_cut'][:,label],\
                             data5['labels_cut'][:,label]))

    features = X_test.shape[1]

    if printInfo:
        print("Training samples: ", X_train.shape[0],\
          "\nTest samples:      ", X_test.shape[0],\
          "\nFeatures:            ", features)

    # decision to overcome the problem of entire missing columns
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    # features normalization
    scaler = StandardScaler().fit(X_train)
    X_train =scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if (make_binary):
        Y_train[Y_train != 0] = 1
        Y_test[Y_test != 0] = 1

    # switch to one hot encoded labels
    onehot_encoder = OneHotEncoder(sparse=False)
    Y_train_oh = onehot_encoder.fit_transform(Y_train.reshape(-1, 1))
    Y_test_oh = onehot_encoder.fit_transform(Y_test.reshape(-1, 1))
    #print("Classes in training set: ", Y_train_oh.shape[1],\
    #  "\nClasses in test set:     ", Y_test_oh.shape[1])

    n_classes = Y_test_oh.shape[1]

    if printInfo:
        print("\nTRAINING SET:")
    X_train_s, Y_train_s = prepareData(X_train, Y_train_oh, window_size, stride, printInfo=printInfo, null_class = null_class)
    if printInfo:
        print("\nTEST SET:")
    X_test_s, Y_test_s = prepareData(X_test, Y_test_oh, window_size, stride, printInfo=printInfo, null_class = null_class)
    
    return (X_train_s, Y_train_s, X_test_s, Y_test_s, n_classes)

def AUC(y_true, y_pred, classes):
    """ Compute the Area Under the Curve of ROC metric. """

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return roc_auc

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def unwindowLabels(Y_s, window_size, stride):
    """ Stretch labels for each window for each temporal sample. """
    
    Y = np.zeros(())
    Y_s.shape[0]

    return Y