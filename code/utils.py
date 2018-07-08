# Auxiliary functions for HDA project on Human Activity Recognition

import numpy as np
import scipy.io
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    from keras import regularizers
    from keras.layers import Conv1D, Conv2D, BatchNormalization, Dropout, LeakyReLU, Flatten, Activation, Dense, MaxPooling1D, MaxPooling2D
    from keras.models import Model, Sequential
    from keras.optimizers import Adam
    import keras.backend as K


def loadData(subject, folder="./"):
    """ Import ADL1 to ADL5 and Drill .mat files for a subject. """
    
    filename_1 = folder + "S" + str(subject) + "-ADL1.mat"
    filename_2 = folder + "S" + str(subject) + "-ADL2.mat"
    filename_3 = folder + "S" + str(subject) + "-ADL3.mat"
    filename_4 = folder + "S" + str(subject) + "-ADL4.mat"
    filename_5 = folder + "S" + str(subject) + "-ADL5.mat"
    filename_6 = folder + "S" + str(subject) + "-Drill.mat"

    data1 = scipy.io.loadmat(filename_1, mdict={'features':'features', 'labels':'labels'})
    data2 = scipy.io.loadmat(filename_2, mdict={'features':'features', 'labels':'labels'})
    data3 = scipy.io.loadmat(filename_3, mdict={'features':'features', 'labels':'labels'})
    data4 = scipy.io.loadmat(filename_4, mdict={'features':'features', 'labels':'labels'})
    data5 = scipy.io.loadmat(filename_5, mdict={'features':'features', 'labels':'labels'})
    data6 = scipy.io.loadmat(filename_6, mdict={'features':'features', 'labels':'labels'})

    print("\nSession shapes:")
    print("ADL1:  ", data1['features'].shape)
    print("ADL2:  ", data2['features'].shape)
    print("ADL3:  ", data3['features'].shape)
    print("ADL4:  ", data4['features'].shape)
    print("ADL5:  ", data5['features'].shape)
    print("Drill: ", data6['features'].shape)

    return (data1, data2, data3, data4, data5, data6)

def prepareData(X, Y, window_size=15, stride=15, shuffle=True):
    """ Prepare data in windows to be passed to the CNN. """

    samples, features = X.shape
    classes = Y.shape[1]
    # shape output
    windows = int(samples // stride) - 1
    X_out = np.zeros([windows, window_size, features])
    Y_out = np.zeros([windows, classes])
    # write output
    for i in range(windows):
        index = int(i * stride)
        X_out[i, :, :] = X[index:index+window_size, :].reshape((window_size,features))
        temp = Y[index:index+window_size, :]
        Y_out[i, np.argmax(np.sum(temp, axis=0))] = 1 # hard version      CHECK!
    print("\nFeatures have shape: ", X_out.shape,\
          "\nLabels have shape:   ", Y_out.shape,\
          "\nFraction of labels:  ", np.sum(Y_out, axis=0) / Y_out.shape[0])

    if shuffle:
        X_out_shuffled = np.random.permutation(X_out)

    return (X_out, Y_out)

def Model1D(input_shape, classes):
    """ 
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- number of classes

    Returns: 
    model -- a Model() instance in Keras
    """
    
    model = Sequential()
    model.add(Conv1D(filters = 18,
                    kernel_size=5,
                    strides=1,
                    padding='same',
                    input_shape = input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2,
                          strides=2,
                          padding='same'))
    
    model.add(Conv1D(filters = 36,
                    kernel_size=7,
                    strides=1,
                    padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2,
                          strides=2,
                          padding='same'))
    
    model.add(Dropout(0.2))
    
    model.add(Conv1D(filters = 72,
                    kernel_size=7,
                    strides=1,
                    padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2,
                          strides=2,
                          padding='same'))
    
    #model.add(Conv1D(filters = 144,
    #                kernel_size=7,
    #                strides=1,
    #                padding='same'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(MaxPooling1D(pool_size=2,
    #                      strides=2,
    #                      padding='same'))
    
    model.add(Flatten())
    
    model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
    model.add(LeakyReLU(alpha=0.3))
    
    model.add(Dropout(0.4))

    model.add(Dense(classes))
    model.add(Activation('softmax'))
    
    model.summary()
    
    return model

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

