# The functions are here organised in three levels because higher levels make use of lower levels functions.

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import class_weight


# low level methods

def readData(subject, folder="./", print_info=False):
    """Read ADL1 to ADL5 and Drill .mat files for a specified subject."""

    filename_1 = folder + "S" + str(subject) + "-ADL1"
    filename_2 = folder + "S" + str(subject) + "-ADL2"
    filename_3 = folder + "S" + str(subject) + "-ADL3"
    filename_4 = folder + "S" + str(subject) + "-ADL4"
    filename_5 = folder + "S" + str(subject) + "-ADL5"
    filename_6 = folder + "S" + str(subject) + "-Drill"

    # load into dictionaries of numpy arrays
    data1 = loadmat(filename_1, mdict={'features_interp':'features', 'labels_cut':'labels'})
    data2 = loadmat(filename_2, mdict={'features_interp':'features', 'labels_cut':'labels'})
    data3 = loadmat(filename_3, mdict={'features_interp':'features', 'labels_cut':'labels'})
    data4 = loadmat(filename_4, mdict={'features_interp':'features', 'labels_cut':'labels'})
    data5 = loadmat(filename_5, mdict={'features_interp':'features', 'labels_cut':'labels'})
    data6 = loadmat(filename_6, mdict={'features_interp':'features', 'labels_cut':'labels'})

    if (print_info):
        print("\nSession shapes:")
        print("ADL1:  ", data1['features_interp'].shape)
        print("ADL2:  ", data2['features_interp'].shape)
        print("ADL3:  ", data3['features_interp'].shape)
        print("ADL4:  ", data4['features_interp'].shape)
        print("ADL5:  ", data5['features_interp'].shape)
        print("Drill: ", data6['features_interp'].shape)

    return (data1, data2, data3, data4, data5, data6)

def shapeData(X, Y, window_size=15, stride=15, null_class = True, printInfo = False):
    """Orgnise data into windows to be passed to the model.
    
    X_out: a 3D numpy array of shape [n_windows, window_size, n_features]
    Y_out: a 2D numpy array, containing one-hot encoded labels, of shape [n_windows, n_classes]
    n_features: integer, number of features
    n_classes: integer, number of classes
    """

    # data shapes
    n_samples, n_features = X.shape
    n_classes = Y.shape[1]

    # format output shape
    n_windows = int(n_samples // stride) - int(window_size // stride) # + 1 # + 2
    X_out = np.zeros([n_windows, window_size, n_features])
    Y_out = np.zeros([n_windows, n_classes])

    # write output
    for i in range(n_windows):
        # compute starting index
        index = int(i * stride)
        # copy window (data and labels) starting from index
        X_out[i, :, :] = X[index:index+window_size, :].reshape((window_size,n_features))
        temp = Y[index:index+window_size, :]
        # use most recurrent label as window class
        Y_out[i, np.argmax(np.sum(temp, axis=0))] = 1

    if not(null_class): # discard windows 0-labeled
        # mask for samples non 0-labeled
        non_null = (Y_out[:,0] == 0) # (Y_out[:,0] is the first column of Y_out)
        # keep only non 0-labeled windows and discard first column of labels
        X_out = X_out[non_null]
        Y_out = Y_out[non_null][:,1:]
        # update n_classes
        n_classes = Y_out.shape[1]

    if (printInfo):
        print("\nFeatures:", n_features,\
              "\nClasses:", n_classes,\
            "\nFraction of labels:  ", np.sum(Y_out, axis=0) / Y_out.shape[0])

    return (X_out, Y_out, n_features, n_classes)


# mid level methods

def loadData(subject, label, folder="./", window_size=15, stride=15, make_binary=False, null_class=True, print_info=False):
    """Preprocess data to be ready for classification models.
    
    X_train_s: 3D numpy array
    Y_train_s: 1D numpy array
    X_test_s: 3D nupy array
    Y_test_s: 1D numpy array
    n_features: integer
    n_classes: integer
    class_weights: 1D numpy array
    """

    # read data
    data1, data2, data3, data4, data5, data6 = readData(subject=subject,
                                                        folder=folder,
                                                        print_info=print_info)

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
                             data5['labels_cut'][:,label]), axis=0)

    # set empty columns (NaNs) to 0
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    # features normalization
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # make the problem binary
    if make_binary:
        Y_train[Y_train != 0] = 1
        Y_test[Y_test != 0] = 1

    # switch to one hot encoded labels
    onehot_encoder = OneHotEncoder(sparse=False)
    Y_train_oh = onehot_encoder.fit_transform(Y_train.reshape(-1, 1))
    Y_test_oh = onehot_encoder.fit_transform(Y_test.reshape(-1, 1))

    # shape data
    X_train_s, Y_train_s, n_features, n_classes = shapeData(X_train,
                                                            Y_train_oh,
                                                            window_size,
                                                            stride,
                                                            null_class=null_class,
                                                            printInfo=print_info)

    X_test_s, Y_test_s, n_features, n_classes = shapeData(X_test,
                                                          Y_test_oh,
                                                          window_size,
                                                          stride,
                                                          null_class=null_class,
                                                          printInfo=print_info)

    # switch labels back to normal 1D array
    Y_train_s = np.argmax(Y_train_s, axis=1)
    Y_test_s = np.argmax(Y_test_s, axis=1)

    # compute class weights
    class_weights = class_weight.compute_class_weight('balanced', np.arange(n_classes), Y_train_s)

    return (X_train_s, Y_train_s, X_test_s, Y_test_s, n_features, n_classes, class_weights)


# high level methods

def loadDataAll(label, folder="./", window_size=15, stride=15, make_binary=False, null_class=True, print_info=False):
    """Process data all subjects to be ready for classification models."""

    print("\nProcessing data from subject 1")
    X_train_1, Y_train_1, X_test_1, Y_test_1, n_features, n_classes = loadData(subject=1, label=label, folder=folder, window_size=window_size, stride=stride,
                                                                               make_binary=make_binary, null_class=null_class, print_info=print_info)[0:6]
    print("\nProcessing data from subject 2")
    X_train_2, Y_train_2, X_test_2, Y_test_2 = loadData(subject=2, label=label, folder=folder, window_size=window_size, stride=stride,
                                                        make_binary=make_binary, null_class=null_class, print_info=print_info)[0:4]
    print("\nProcessing data from subject 3")
    X_train_3, Y_train_3, X_test_3, Y_test_3 = loadData(subject=3, label=label, folder=folder, window_size=window_size, stride=stride,
                                                        make_binary=make_binary, null_class=null_class, print_info=print_info)[0:4]
    print("\nProcessing data from subject 4")
    X_train_4, Y_train_4, X_test_4, Y_test_4 = loadData(subject=4, label=label, folder=folder, window_size=window_size, stride=stride,
                                                        make_binary=make_binary, null_class=null_class, print_info=print_info)[0:4]

    # create training set and test set
    X_train = np.concatenate((X_train_1,\
                              X_train_2,\
                              X_train_3,\
                              X_train_4), axis=0)

    Y_train = np.concatenate((Y_train_1,\
                              Y_train_2,\
                              Y_train_3,\
                              Y_train_4), axis=0)

    X_test = np.concatenate((X_test_1,\
                             X_test_2,\
                             X_test_3,\
                             X_test_4), axis=0)

    Y_test = np.concatenate((Y_test_1,\
                             Y_test_2,\
                             Y_test_3,\
                             Y_test_4), axis=0)

    if print_info:
        print("\nShapes:",
              "\nX_train: ", X_train.shape,
              "\nY_train: ", Y_train.shape,
              "\nX_test: ", X_test.shape,
              "\nY_test: ", Y_test.shape)

    # class weights
    class_weights = class_weight.compute_class_weight('balanced', np.arange(n_classes), Y_train)
    if print_info:
        print("\nClass weights:\n", class_weights)

    return X_train, Y_train, X_test, Y_test, n_features, n_classes, class_weights

def loadDataMultiple(label, folder="./", window_size=15, stride=15, make_binary=False, null_class=True, print_info=False):
    """Process data of a list of subjects to be ready for classification models."""

    #print("\nProcessing data from subject 1")
    #X_train_1, Y_train_1, X_test_1, Y_test_1,  = loadData(subject=1, label=label, folder=folder, window_size=window_size, stride=stride,
    #                                                                           make_binary=make_binary, null_class=null_class, print_info=print_info)[0:4]
    print("\nProcessing data from subject 2")
    X_train_2, Y_train_2, X_test_2, Y_test_2, n_features, n_classes = loadData(subject=2, label=label, folder=folder, window_size=window_size, stride=stride,
                                                                               make_binary=make_binary, null_class=null_class, print_info=print_info)[0:6]
    print("\nProcessing data from subject 3")
    X_train_3, Y_train_3, X_test_3, Y_test_3 = loadData(subject=3, label=label, folder=folder, window_size=window_size, stride=stride,
                                                        make_binary=make_binary, null_class=null_class, print_info=print_info)[0:4]
    #print("\nProcessing data from subject 4")
    #X_train_4, Y_train_4, X_test_4, Y_test_4 = loadData(subject=4, label=label, folder=folder, window_size=window_size, stride=stride,
    #                                                    make_binary=make_binary, null_class=null_class, print_info=print_info)[0:4]

    # create training set and test set
    X_train = np.concatenate((X_train_2,\
                              X_train_3), axis=0)

    Y_train = np.concatenate((Y_train_2,\
                              Y_train_3), axis=0)

    X_test = np.concatenate((X_test_2,\
                             X_test_3), axis=0)

    Y_test = np.concatenate((Y_test_2,\
                             Y_test_3), axis=0)

    if print_info:
        print("\nShapes:",
              "\nX_train: ", X_train.shape,
              "\nY_train: ", Y_train.shape,
              "\nX_test: ", X_test.shape,
              "\nY_test: ", Y_test.shape)

    # class weights
    class_weights = class_weight.compute_class_weight('balanced', np.arange(n_classes), Y_train)
    if print_info:
        print("\nClass weights:\n", class_weights)

    return X_train, Y_train, X_test, Y_test, n_features, n_classes, class_weights