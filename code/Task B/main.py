import preprocessing
import models
import utils
import numpy as np
from sklearn.metrics import classification_report, f1_score
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical

# PARAMETERS #####################################################################
subject = 1
label = 6   # default for task B1
folder = "../data/full/"
window_size = 15
stride = 5
##################################################################################

# create folder to store temporary data
import os
if not(os.path.exists("./data")):
    os.mkdir("./data")

# PREPROCESSING
X_train, Y_train, X_test, Y_test, n_features, n_classes, class_weights = preprocessing.loadData(subject=subject,
                                                                                                label=label,
                                                                                                folder=folder,
                                                                                                window_size=window_size,
                                                                                                stride=stride,
                                                                                                make_binary=False,
                                                                                                null_class=True,
                                                                                                print_info=True)

# create folder to store temporary data
import os
if not(os.path.exists("./data")):
    os.mkdir("./data")

# PREPROCESSING
X_train, Y_train, X_test, Y_test, n_features, n_classes, class_weights = preprocessing.loadData(subject=subject,
                                                                                                label=label,
                                                                                                folder=folder,
                                                                                                window_size=window_size,
                                                                                                stride=stride,
                                                                                                make_binary=False,
                                                                                                null_class=True,
                                                                                                print_info=True)

# MODEL
model_name = "Convolutional" # keep update for automatic name to store the model
model = models.Convolutional((window_size, n_features), n_classes, print_info=True)

model.compile(optimizer = Adam(lr=0.001),
              loss = "categorical_crossentropy", 
              metrics = ["accuracy"])

checkpointer = ModelCheckpoint(filepath='./data/model_terminal.hdf5', verbose=1, save_best_only=True)
lr_reducer = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)

# TRAINING
model.fit(x = X_train, 
          y = to_categorical(Y_train), 
          epochs = 10, 
          batch_size = 64,
          verbose = 1,
          validation_data=(X_test, to_categorical(Y_test)),
          callbacks=[checkpointer, lr_reducer])

# EVALUATION - Da sistemare
print("Last model:\n")
Y_pred = model.predict_classes(X_test)
print(classification_report(Y_test, Y_pred))
print("Weighted f1-score:", f1_score(Y_test, Y_pred, average='weighted'))

print("Best model:\n")
model_best = load_model('./data/model_terminal.hdf5')
Y_pred = model_best.predict_classes(X_test)
print(classification_report(Y_test, Y_pred))
print("Weighted f1-score:", f1_score(Y_test, Y_pred, average='weighted'))