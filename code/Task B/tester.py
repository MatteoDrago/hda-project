import preprocessing
import models
import numpy as np
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

# PARAMETERS #####################################################################
subject = 1
label = 6   # default for task B1
folder = "../data/full/"
window_size = 15
stride = 5
##################################################################################

# PREPROCESSING
X_train, Y_train, X_test, Y_test, n_features, n_classes, class_weights = preprocessing.loadData(subject=subject,
                                                                                                        label=label,
                                                                                                        folder=folder,
                                                                                                        window_size=window_size,
                                                                                                        stride=stride,
                                                                                                        make_binary=False,
                                                                                                        null_class=True,
                                                                                                        print_info=True)

# reshaping for 2D convolutional model
X_train = X_train.reshape(X_train.shape[0], window_size, n_features, 1)
X_test = X_test.reshape(X_test.shape[0], window_size, n_features, 1)

# MODEL
detection_model = models.MotionClassification2D((window_size, n_features, 1), n_classes, print_info=True)

detection_model.compile(optimizer = Adam(lr=0.0015),
                        loss = "categorical_crossentropy", 
                        metrics = ["accuracy"])

checkpointer = ModelCheckpoint(filepath='./weights_terminal.hdf5', verbose=1, save_best_only=True)

# TRAINING
detection_model.fit(x = X_train, 
                    y = to_categorical(Y_train), 
                    epochs = 10, 
                    batch_size = 256,
                    verbose = 1,
                    validation_data=(X_test, to_categorical(Y_test)),
                    callbacks=[checkpointer])

# EVALUATION
print("Last weights:\n")
Y_pred = detection_model.predict_classes(X_test)
print(classification_report(Y_test, Y_pred))

print("Best weights:\n")
detection_model_best = load_model('./weights_terminal.hdf5')
Y_pred = detection_model.predict_classes(X_test)
print(classification_report(Y_test, Y_pred))