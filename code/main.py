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

# PARAMETERS ########################################################################################################
subject = [1,2,3,4]
task = "A"    # choose between "A" or "B"
model_names = ["Convolutional", "Convolutional1DRecurrent", "Convolutional2DRecurrent", "ConvolutionalDeepRecurrent"]
data_folder = "../data/full/"
window_size = 15
stride = 5
GPU = True
epochs = 10
batch_size = 32
#####################################################################################################################

# create folder to store temporary data and results
if not(os.path.exists("./data")):
    os.mkdir("./data")
if not(os.path.exists("./data/models")):
    os.mkdir("./data/models")
if not(os.path.exists("./data/results")):
    os.mkdir("./data/results")

# select labels according to the task selected
if task == "A":
    label = 0
elif task == "B":
    label = 6
else:
    print("Error: invalid task.")

# the program is repeated for all models in the list
for model_name in model_names:

    # and for all subjects in the list
    for s in subject:
        print("\n\nSubcject", str(s))


        # ONE-SHOT CLASSIFICATION
        print("One-shot classification")

        # preprocessing
        X_train, Y_train, X_test, Y_test, n_features, n_classes, class_weights = preprocessing.loadData(subject=s,
                                                                                                        label=label,
                                                                                                        folder=data_folder,
                                                                                                        window_size=window_size,
                                                                                                        stride=stride,
                                                                                                        make_binary=False,
                                                                                                        null_class=True,
                                                                                                        print_info=False)

        # model
        if model_name == "Convolutional":
            model = models.Convolutional((window_size, n_features), n_classes, print_info=False)
        elif model_name == "Convolutional1DRecurrent":
            model = models.Convolutional1DRecurrent((window_size, n_features), n_classes, GPU=GPU, print_info=False)
        elif model_name == "Convolutional2DRecurrent":
            model = models.Convolutional2DRecurrent((window_size, n_features, 1), n_classes, GPU=GPU, print_info=False)
            # reshaping for 2D convolutional model
            X_train = X_train.reshape(X_train.shape[0], window_size, n_features, 1)
            X_test = X_test.reshape(X_test.shape[0], window_size, n_features, 1)
        elif model_name == "ConvolutionalDeepRecurrent":
            model = models.ConvolutionalDeepRecurrent((window_size, n_features), n_classes, GPU=GPU, print_info=False)
        else:
            print("Model not found.")
            break
        model.compile(optimizer = Adam(lr=0.001), loss = "categorical_crossentropy", metrics = ["accuracy"])
        save_model_name = task + "_" + model_name + "_OS_" + str(s)
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

        # results
        # last model
        Y_pred = model.predict_classes(X_test)
        score_OS = f1_score(Y_test, Y_pred, average='weighted')
        # best model
        model_best = load_model(filepath)
        Y_pred_best = model_best.predict_classes(X_test)
        score_OS_best = f1_score(Y_test, Y_pred_best, average='weighted')
        # keep highest f1-score
        if score_OS_best > score_OS:
            score_OS = score_OS_best
            print("\nResults for best "+ save_model_name + ":\n", classification_report(Y_test, Y_pred))
        else:
            print("\nResults for last "+ save_model_name + ":\n", classification_report(Y_test, Y_pred))


        # TWO-STEPS CLASSIFICATION - DETECTION
        print("Two-steps classification - detecion")

        # preprocessing
        X_train, Y_train, X_test, Y_test, n_features, n_classes, class_weights = preprocessing.loadData(subject=s,
                                                                                                        label=label,
                                                                                                        folder=data_folder,
                                                                                                        window_size=window_size,
                                                                                                        stride=stride,
                                                                                                        make_binary=True,
                                                                                                        null_class=True,
                                                                                                        print_info=False)

        # model
        if model_name == "Convolutional":
            model = models.Convolutional((window_size, n_features), n_classes, print_info=False)
        elif model_name == "Convolutional1DRecurrent":
            model = models.Convolutional1DRecurrent((window_size, n_features), n_classes, GPU=GPU, print_info=False)
        elif model_name == "Convolutional2DRecurrent":
            model = models.Convolutional2DRecurrent((window_size, n_features, 1), n_classes, GPU=GPU, print_info=False)
            # reshaping for 2D convolutional model
            X_train = X_train.reshape(X_train.shape[0], window_size, n_features, 1)
            X_test = X_test.reshape(X_test.shape[0], window_size, n_features, 1)
        elif model_name == "ConvolutionalDeepRecurrent":
            model = models.ConvolutionalDeepRecurrent((window_size, n_features), n_classes, GPU=GPU, print_info=False)
        else:
            print("Model not found.")
            break
        model.compile(optimizer = Adam(lr=0.001), loss = "categorical_crossentropy", metrics = ["accuracy"])
        save_model_name = task + "_" + model_name + "_TSD_" + str(s)
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

        # results
        # last model
        Y_pred = model.predict_classes(X_test)
        score_TSD = f1_score(Y_test, Y_pred, average='weighted')
        # best model
        model_best = load_model(filepath)
        Y_pred_best = model_best.predict_classes(X_test)
        score_TSD_best = f1_score(Y_test, Y_pred_best, average='weighted')
        # keep highest f1-score
        if score_TSD_best > score_TSD:
            score_TSD = score_TSD_best
            print("\nResults for best "+ save_model_name + ":\n", classification_report(Y_test, Y_pred))
        else:
            print("\nResults for last "+ save_model_name + ":\n", classification_report(Y_test, Y_pred))

        
        # TWO-STEPS CLASSIFICATION - CLASSIFICATION
        print("Two-steps classification - classification")

        # preprocessing
        X_train, Y_train, X_test, Y_test, n_features, n_classes, class_weights = preprocessing.loadData(subject=s,
                                                                                                        label=label,
                                                                                                        folder=data_folder,
                                                                                                        window_size=window_size,
                                                                                                        stride=stride,
                                                                                                        make_binary=False,
                                                                                                        null_class=False,
                                                                                                        print_info=False)

        # model
        if model_name == "Convolutional":
            model = models.Convolutional((window_size, n_features), n_classes, print_info=False)
        elif model_name == "Convolutional1DRecurrent":
            model = models.Convolutional1DRecurrent((window_size, n_features), n_classes, GPU=GPU, print_info=False)
        elif model_name == "Convolutional2DRecurrent":
            model = models.Convolutional2DRecurrent((window_size, n_features, 1), n_classes, GPU=GPU, print_info=False)
            # reshaping for 2D convolutional model
            X_train = X_train.reshape(X_train.shape[0], window_size, n_features, 1)
            X_test = X_test.reshape(X_test.shape[0], window_size, n_features, 1)
        elif model_name == "ConvolutionalDeepRecurrent":
            model = models.ConvolutionalDeepRecurrent((window_size, n_features), n_classes, GPU=GPU, print_info=False)
        else:
            print("Model not found.")
            break
        model.compile(optimizer = Adam(lr=0.001), loss = "categorical_crossentropy", metrics = ["accuracy"])
        save_model_name = task + "_" + model_name + "_TSC_" + str(s)
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

        # results
        # last model
        Y_pred = model.predict_classes(X_test)
        score_TSC = f1_score(Y_test, Y_pred, average='weighted')
        # best model
        model_best = load_model(filepath)
        Y_pred_best = model_best.predict_classes(X_test)
        score_TSC_best = f1_score(Y_test, Y_pred_best, average='weighted')
        # keep highest f1-score
        if score_TSC_best > score_TSC:
            score_TSC = score_TSC_best
            print("\nResults for best "+ save_model_name + ":\n", classification_report(Y_test, Y_pred))
        else:
            print("\nResults for last "+ save_model_name + ":\n", classification_report(Y_test, Y_pred))

        # store results
        np.savetxt("./data/results/"+task+"_"+model_name+"_"+str(s)+".txt", [score_OS, score_TSD, score_TSC], fmt="%1.4f")  # try saving as text