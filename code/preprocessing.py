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

    # switch to one hot encoded labels
    onehot_encoder = OneHotEncoder(sparse=False)
    Y = onehot_encoder.fit_transform(Y.reshape(-1, 1))

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
    
    # switch labels back to normal 1D array
    Y_out = np.argmax(Y_out, axis=1)

    return (X_out, Y_out, n_features, n_classes)


# mid level methods

def loadData(subject, label, folder="./", window_size=15, stride=15, make_binary=False, null_class=True, balcance_classes=False, print_info=False):
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
    data1, data2, data3, data4, data5, data6 = readData(subject=subject, folder=folder, print_info=print_info)

    # create training set and test sets
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

    # shape data
    X_train, Y_train, n_features, n_classes = shapeData(X_train, Y_train, window_size, stride, null_class=null_class, printInfo=print_info)
    X_test, Y_test, n_features, n_classes = shapeData(X_test, Y_test, window_size, stride, null_class=null_class, printInfo=print_info)    

    # balance data
    if balcance_classes:
        class_weights = class_weight.compute_class_weight('balanced', np.arange(n_classes), Y_train)
        class_weights = np.floor(class_weights / np.min(class_weights))
        class_weights
        elenco = []
        elenco2 = []
        for l in enumerate(class_weights):
            print("\n",l)
            mask = (Y_train == l[0])
            print(mask)
            print(X_train[mask,:,:].shape, Y_train[mask].shape)
            X_train_bal = np.tile(X_train[mask,:,:], (int(l[1]),1,1))
            Y_train_bal = np.tile(Y_train[mask], int(l[1]))
            print(X_train_bal.shape, Y_train_bal.shape)
            for i in range(X_train_bal.shape[0]):
                elenco.append(X_train_bal[i,:,:] + np.random.normal(scale=0.1)) # add here normal distributed noise
                elenco2.append(Y_train_bal[i])
            print(len(elenco), len(elenco2))
        X_train = np.asarray(elenco)
        print(X_train.shape, type(X_train))
        del elenco
        Y_train = np.asarray(elenco2)
        print(Y_train.shape, type(Y_train))
        del elenco2
        rng_state = np.random.get_state()
        np.random.shuffle(X_train)
        np.random.set_state(rng_state)
        np.random.shuffle(Y_train)

    return X_train, Y_train, X_test, Y_test, n_features, n_classes


# high level methods

def loadDataAll(label, folder="./", window_size=15, stride=15, make_binary=False, null_class=True, balcance_classes=False, print_info=False):
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

    return X_train, Y_train, X_test, Y_test, n_features, n_classes

def loadDataMultiple(label, folder="./", window_size=15, stride=15, make_binary=False, null_class=True, balcance_classes=False, print_info=False):
    """Process data of a list of subjects to be ready for classification models."""

    if print_info:
        print("Processing data from subject 2")
    X_train_2, Y_train_2, X_test_2, Y_test_2, n_features, n_classes = loadData(subject=2, label=label, folder=folder, window_size=window_size, stride=stride,
                                                                               make_binary=make_binary, null_class=null_class,
                                                                               balcance_classes=balcance_classes, print_info=print_info)[0:6]
    if print_info:
        print("Processing data from subject 3")
    X_train_3, Y_train_3, X_test_3, Y_test_3 = loadData(subject=3, label=label, folder=folder, window_size=window_size, stride=stride,
                                                        make_binary=make_binary, null_class=null_class,
                                                        balcance_classes=balcance_classes, print_info=print_info)[0:4]

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

    return X_train, Y_train, X_test, Y_test, n_features, n_classes