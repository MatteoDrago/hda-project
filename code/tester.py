import preprocessing
import models
import utils
import launch
import os
import numpy as np
from sklearn.metrics import classification_report, f1_score
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical

# PARAMETERS ########################################################################################################
subjects = [1,2,3,4]
tasks = ["B"]
model_names = ["ConvolutionalDeepRecurrent"]#["Convolutional", "Convolutional1DRecurrent", "Convolutional2DRecurrent", "ConvolutionalDeepRecurrent"]
data_folder = "./data/full/"
window_size = 10
stride = 5
GPU = True
epochs = 5
batch_size = 32
#####################################################################################################################

# create folder to store temporary data and results
if not(os.path.exists("./data")):
    os.mkdir("./data")
if not(os.path.exists("./data/models")):
    os.mkdir("./data/models")
if not(os.path.exists("./data/results")):
    os.mkdir("./data/results")

for task in tasks:
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
        for subject in subjects:
            print("\n\nSubcject", str(subject))


            # ONE-SHOT CLASSIFICATION
            print("One-shot classification")

            model, X_test, Y_test, filepath, save_model_name, n_features = launch.oneshot_classification(subject, task, model_name, data_folder,
                                                                                                         window_size=window_size, stride=stride, epochs=epochs,
                                                                                                         batch_size=batch_size, GPU=GPU, print_info=False)

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
                print("\nResults for best "+ save_model_name + ":\n", classification_report(Y_test, Y_pred_best))
                # save for future use
                Y_true = Y_test
                Y_OS = Y_pred_best
            else:
                print("\nResults for last "+ save_model_name + ":\n", classification_report(Y_test, Y_pred))
                # save for future use
                Y_true = Y_test
                Y_OS = Y_pred


            # TWO-STEPS CLASSIFICATION - DETECTION
            print("Two-steps classification - detecion")

            model, X_test, Y_test, filepath, save_model_name, n_features = launch.cascade_detection(subject, task, model_name, data_folder,
                                                                                                    window_size=window_size, stride=stride, epochs=epochs,
                                                                                                    batch_size=batch_size, GPU=GPU, print_info=False)

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
                print("\nResults for best "+ save_model_name + ":\n", classification_report(Y_test, Y_pred_best))
                # save for future use
                Y_det = Y_pred_best
            else:
                print("\nResults for last "+ save_model_name + ":\n", classification_report(Y_test, Y_pred))
                # save for future use
                Y_det = Y_pred

            
            # TWO-STEPS CLASSIFICATION - CLASSIFICATION
            print("Two-steps classification - classification")

            model, X_test, Y_test, filepath, save_model_name, n_features = launch.cascade_classification(subject, task, model_name, data_folder,
                                                                                                         window_size=window_size, stride=stride, epochs=epochs,
                                                                                                         batch_size=batch_size, GPU=GPU, print_info=False)

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
                print("\nResults for best "+ save_model_name + ":\n", classification_report(Y_test, Y_pred_best))
            else:
                print("\nResults for last "+ save_model_name + ":\n", classification_report(Y_test, Y_pred))

            
            # CASCADE (Two Steps classification)

            # get test set
            X_test = preprocessing.loadData(subject=subject,
                                            label=label,
                                            folder=data_folder,
                                            window_size=window_size,
                                            stride=stride,
                                            make_binary=False,
                                            null_class=True,
                                            print_info=False)[2]

            # mask
            mask = (Y_det == 1)
            activity_windows = X_test[mask, :, :]
            if model_name == "Convolutional2DRecurrent":
                activity_windows = activity_windows.reshape(activity_windows.shape[0], window_size, n_features, 1)
            Y_clas = model_best.predict_classes(activity_windows) + 1
            Y_TS = Y_det
            Y_TS[mask] = Y_clas
            score_TS = f1_score(Y_true, Y_TS, average='weighted')
            print("Two-Steps results:\n", classification_report(Y_true, Y_TS))

            # store results
            np.savetxt("./data/results/"+task+"_"+model_name+"_"+str(subject)+".txt", [score_OS, score_TSD, score_TSC, score_TS], fmt="%1.4f")  # try saving as text