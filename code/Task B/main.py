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

# MODEL
model = models.MotionDetection((window_size, n_features), n_classes, print_info=True)

model.compile(optimizer = Adam(lr=0.001),
              loss = "categorical_crossentropy", 
              metrics = ["accuracy"])

checkpointer = ModelCheckpoint(filepath='./model_terminal.hdf5', verbose=1, save_best_only=True)

# TRAINING
model.fit(x = X_train, 
          y = to_categorical(Y_train), 
          epochs = 20, 
          batch_size = 16,
          verbose = 1,
          validation_data=(X_test, to_categorical(Y_test)),
          callbacks=[checkpointer])

# EVALUATION - Da sistemare
print("Last weights:\n")
Y_pred = model.predict(X_test)
Y_pred = np.argmax(Y_pred, 1)
print(classification_report(Y_test, to_categorical(Y_pred)))

print("Best weights:\n")
model_best = load_model('./model_terminal.hdf5')
Y_pred = model_best.predict(X_test)
Y_pred = np.argmax(Y_pred, 1)
print(classification_report(Y_test, to_categorical(Y_pred)))