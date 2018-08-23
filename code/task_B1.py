import utils
import deeplearning
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from keras.optimizers import Adam

# PARAMETERS #####################################################################
subject = 1
folder = "./data/full/"
label = 6     # default for task B1
window_size = 15
stride = 5
# make_binary = True
##################################################################################

[x_train, y_train, x_test, y_test] = utils.preprocessing(subject,
                                                                    folder,
                                                                    label,
                                                                    window_size,
                                                                    stride,
                                                                    printInfo = True,
                                                                    make_binary = True)

n_features = 110 #number of features taken into consideration for the solution of the problem
n_classes = 2

detection_model = deeplearning.MotionDetection((window_size,n_features,1), n_classes)
detection_model.summary() # model visualization

detection_model.compile(optimizer = Adam(lr=0.01), 
                        loss = "categorical_crossentropy", 
                        metrics = ["accuracy"])

input_train = x_train.reshape(x_train.shape[0], window_size, n_features, 1)
input_test = x_test.reshape(x_test.shape[0], window_size, n_features, 1)

detection_model.fit(x = input_train, 
                    y = y_train, 
                    epochs = 20, 
                    batch_size = 300,
                    verbose = 1,
                    validation_data=(input_test, y_test))

y_pred = detection_model.predict(input_test)

print(classification_report(y_test, y_pred))