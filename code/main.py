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
subjects = [1,2,3,4,23]
tasks = ["A","B"]
model_names = ["Convolutional", "Convolutional1DRecurrent", "Convolutional2DRecurrent", "ConvolutionalDeepRecurrent"]
data_folder = "./data/full/"
window_size = 10
stride = 5
GPU = True
epochs = 5
batch_size = 32
balcance_classes = False
print_info = False
#####################################################################################################################

# create folder to store temporary data and results
if not(os.path.exists("./data")):
    os.mkdir("./data")
if not(os.path.exists("./data/models")):
    os.mkdir("./data/models")
if not(os.path.exists("./data/results")):
    os.mkdir("./data/results")

# loop through tasks, models and subjects
for task in tasks:

    # select labels according to the task selected
    if task == "A":
        label = 0
    elif task == "B":
        label = 6
    else:
        print("Error: invalid task.")

    for model_name in model_names:

        for subject in subjects:

            # ONE-SHOT CLASSIFICATION
            print("\n\nCurrent configuration:  Task " + task + ";  Model " + model_name + ";  Subject", str(subject))
            print("One-shot classification")
            # preprocessing, model creation and training
            model, X_test, Y_test, filepath, save_model_name = launch.oneshot_classification(subject, task, model_name, data_folder,
                                                                                             window_size=window_size, stride=stride, epochs=epochs,
                                                                                             batch_size=batch_size, balcance_classes=False,
                                                                                             GPU=GPU, print_info=print_info)
            # results
            Y_pred_os, score_os = launch.evaluation(model, X_test, Y_test, filepath, save_model_name)
            # save for future use
            Y_true = Y_test


            # ACTIVITY DETECTION
            print("\n\nCurrent configuration:  Task " + task + ";  Model " + model_name + ";  Subject", str(subject))
            print("Activity detecion")
            # preprocessing, model creation and training
            model, X_test, Y_test, filepath, save_model_name = launch.cascade_detection(subject, task, model_name, data_folder,
                                                                                        window_size=window_size, stride=stride, epochs=epochs,
                                                                                        batch_size=batch_size, balcance_classes=balcance_classes,
                                                                                        GPU=GPU, print_info=False)
            # results
            Y_pred_ad, score_ad = launch.evaluation(model, X_test, Y_test, filepath, save_model_name)

            
            # ACTIVITY CLASSIFICATION
            print("\n\nCurrent configuration:  Task " + task + ";  Model " + model_name + ";  Subject", str(subject))
            print("Activity classification")
            # preprocessing, model creation and training
            model, X_test, Y_test, filepath, save_model_name = launch.cascade_classification(subject, task, model_name, data_folder,
                                                                                             window_size=window_size, stride=stride, epochs=epochs,
                                                                                             batch_size=batch_size, balcance_classes=balcance_classes,
                                                                                             GPU=GPU, print_info=False)
            # results
            Y_pred_ac, score_ac = launch.evaluation(model, X_test, Y_test, filepath, save_model_name)

            
            # CASCADE: DETECTION + CLASSIFICATION
            print("\n\nCurrent configuration:  Task " + task + ";  Model " + model_name + ";  Subject", str(subject))
            print("Cascade: detecion + classification")
            # get test set
            if subject == 23:
                X_test = preprocessing.loadDataMultiple(label=label,
                                                        folder=data_folder,
                                                        window_size=window_size,
                                                        stride=stride,
                                                        make_binary=False,
                                                        null_class=True,
                                                        print_info=print_info)[2]
            else:
                X_test = preprocessing.loadData(subject=subject,
                                                label=label,
                                                folder=data_folder,
                                                window_size=window_size,
                                                stride=stride,
                                                make_binary=False,
                                                null_class=True,
                                                print_info=print_info)[2]
            # mask
            mask = (Y_pred_ad == 1)
            activity_windows = X_test[mask, :, :]
            if model_name == "Convolutional2DRecurrent":
                activity_windows = activity_windows.reshape(activity_windows.shape[0], window_size, X_test.shape[1], 1)
            Y_casc_ac = model.predict_classes(activity_windows) + 1  # last model saved is "activity classification"
            Y_casc = Y_pred_ad
            Y_casc[mask] = Y_casc_ac
            score_casc = f1_score(Y_true, Y_casc, average='weighted')
            print("Two-Steps results:\n", classification_report(Y_true, Y_casc))

            # store results as text
            np.savetxt("./data/results/"+task+"_"+model_name+"_"+str(subject)+".txt", [score_os, score_ad, score_ac, score_casc], fmt="%1.4f")