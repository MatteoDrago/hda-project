from tqdm import tqdm # progress bar in loop cycles

import numpy as np

from sklearn.svm import LinearSVC

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Merge, Add, merge
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional


def MotionDetection(input_shape, classes, withSoftmax = True):
    
    model = Sequential()
  
    # Layer 0
    model.add(BatchNormalization(input_shape = input_shape))

    # Layer 1
    model.add(Conv2D(filters = 50,
                    kernel_size = (11,3),
                    strides=(1,1),
                    activation='relu'))
    
    # Layer 2
    model.add(MaxPooling2D(pool_size=(2,1)))
    
    # Layer 3
    # This layer dimension are automatically scanned in order to avoid updating by hand each time
    model.add(Reshape((model.layers[2].output_shape[1],model.layers[2].output_shape[2] * model.layers[2].output_shape[3])))  

    # Layer 4
    model.add(LSTM(20,
                  return_sequences=True))
    
    # Layer 5 
    model.add(LSTM(20))
   
    # Layer 6
    model.add(Dense(512,activation = 'relu'))
    
    if (withSoftmax):
        # Layer 7
        model.add(Dense(classes, activation = 'softmax'))
    
    return model

#-------------------------------------------------------------------------------------------------------
# MotionClassification: define a batch normalization + convolutional/max-pooling + 2 LSTM layers DNN
#-------------------------------------------------------------------------------------------------------
def MotionClassification(input_shape, classes, withSoftmax = True):
    
    model = Sequential()
  
    # Layer 0
    model.add(BatchNormalization(input_shape = input_shape))

    # Layer 1
    model.add(Conv2D(filters = 50,
                    kernel_size = (11,1),
                    activation='relu'))
    
    #model.add(MaxPooling2D(pool_size=(2,1)))

    # Layer 2
    model.add(BatchNormalization())

    # Layer 3
    model.add(Conv2D(filters = 50,
                    kernel_size = (11,1),
                    activation='relu'))
    
    # Layer 4
    model.add(MaxPooling2D(pool_size=(2,1)))
    
    # Layer 5
    model.add(BatchNormalization())
    
    # Layer 6
    # This layer dimension are automatically scanned in order to avoid updating by hand each time
    model.add(Reshape((model.layers[5].output_shape[1],model.layers[5].output_shape[2] * model.layers[5].output_shape[3])))  

    # Layer 7
    model.add(LSTM(300,
                  return_sequences=True))
    
    # Layer 8 
    model.add(LSTM(300))
   
    # Layer 9
    model.add(Dropout(0.5))

    # Layer 10
    model.add(Dense(512,activation = 'relu'))

    # Layer 11
    model.add(Dropout(0.5))
    
    if (withSoftmax):
        # Layer 12
        model.add(Dense(classes, activation = 'softmax'))
    
    return model
    
def extractFeatures(model, x_train, x_test, featureSize, batchSize = 600):

	trainingShape =  x_train.shape
	testingShape = x_test.shape

	# Dimension definition
	nbTrainingExamples = trainingShape[0]
	nbTestingExamples = testingShape[0]

	# Allocate the feature arrays
	trainingDnnFeatures = np.empty((nbTrainingExamples,featureSize),dtype=np.float32)
	testingDnnFeatures = np.empty((nbTestingExamples,featureSize),dtype=np.float32)

	print('Computing DNN features on the training set...')
	idx = 0
	iterations = int (nbTrainingExamples // batchSize)
	for i in tqdm(range(iterations+1),ncols=100,ascii=True,desc="TRAINING"):
	    if idx + batchSize < nbTrainingExamples:
	        endIdx = idx+batchSize
	        size = batchSize
	    else:
	        endIdx = nbTrainingExamples
	        size = nbTrainingExamples-idx
	    predictions = model.predict(x_train[idx:endIdx],batch_size=size)
	    trainingDnnFeatures[idx:endIdx] = predictions
	    idx += batchSize

	print('Computing DNN features on the testing set...')
	idx = 0
	iterations = int (nbTestingExamples // batchSize)
	for i in tqdm(range(iterations+1),ncols=100,ascii=True,desc="TESTING"):
	    if idx + batchSize < nbTestingExamples:
	        endIdx = idx+batchSize
	        size = batchSize
	    else:
	        endIdx = nbTestingExamples
	        size = nbTestingExamples-idx
	    predictions = model.predict(x_test[idx:endIdx],batch_size=size)
	    testingDnnFeatures[idx:endIdx] = predictions
	    idx += batchSize

	return (trainingDnnFeatures, testingDnnFeatures)

def SVMLayer(C, y_train, trainingDnnFeatures, testingDnnFeatures):

	for idx in range(len(C)):

	    print('Training the model with C = %.4f ...' % (C[idx]))

	    classifier = LinearSVC(C=C[idx])
	    classifier.fit(trainingDnnFeatures,y_train)
	    estimatedLabels = classifier.predict(testingDnnFeatures)

	return estimatedLabels

