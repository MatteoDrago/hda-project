import utils
import deeplearning
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# PARAMETERS #####################################################################
subject = 1
folder = "./data/full/"
label = 6     # default for task B1
window_size = 15
stride = 5
##################################################################################

# PREPROCESSING
[x_train, y_train, x_test, y_test, n_features, n_classes] = utils.preprocessing(subject,
                                                                                folder,
                                                                                label,
                                                                                window_size,
                                                                                stride,
                                                                                printInfo=True,
                                                                                make_binary=False,
                                                                                null_class=True)

# MODEL
classification_model = deeplearning.MotionClassification((window_size,n_features,1), n_classes)
classification_model.summary() # model visualization

classification_model.compile(optimizer = Adam(lr=0.01),
                             loss = "categorical_crossentropy", 
                             metrics = ["accuracy"])

input_train = x_train.reshape(x_train.shape[0], window_size, n_features, 1)
input_test = x_test.reshape(x_test.shape[0], window_size, n_features, 1)

# TRAINING
classification_model.fit(x = input_train, 
                         y = y_train, 
                         epochs = 20, 
                         batch_size = 16,
                         verbose = 1,
                         validation_data=(input_test, y_test))

# TEST
y_pred = classification_model.predict(input_test)

# RESULTS
print(classification_report(y_test, y_pred))