from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Merge, Add, merge, Conv1D, MaxPooling1D, LeakyReLU, CuDNNLSTM, CuDNNGRU
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.regularizers import l2

# each of the following methods builds and returns a keras model

def Convolutional(input_shape, n_classes, print_info = False):
    """5-layer neural network, with 3 convolutional and 2 fully connected layers."""

    model = Sequential()

    # Layer 1
    model.add(BatchNormalization(input_shape = input_shape))
    model.add(Conv1D(filters = 18, kernel_size=5, strides=1, padding='same'))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    
    # Layer 2
    model.add(BatchNormalization())
    model.add(Conv1D(filters = 36, kernel_size=7, strides=1, padding='same'))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    
    # Layer 3
    model.add(BatchNormalization())
    model.add(Conv1D(filters = 72, kernel_size=7, strides=1, padding='same'))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    
    model.add(Flatten())
    
    # Layer 4
    model.add(Dense(64, kernel_regularizer=l2(0.01)))
    model.add(LeakyReLU(alpha=0.3))
    
    #model.add(Dropout(0.4))

    # Layer 5
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    
    if print_info:
        model.summary()
    
    return model

def Convolutional2DRecurrent(input_shape, n_classes, GPU=False, print_info = False):
    """5-layer neural network, with 1 2D convolutional and 2 LSTM and 2 fully connected layers."""

    model = Sequential()
  
    # Layer 1
    model.add(BatchNormalization(input_shape = input_shape))
    model.add(Conv2D(filters = 50, kernel_size = (11,1), activation='relu'))
    #model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2,1)))
        
    # This layer dimension are automatically scanned in order to avoid updating by hand each time
    model.add(Reshape((model.layers[2].output_shape[1],model.layers[2].output_shape[2] * model.layers[2].output_shape[3])))  

    # Layer 2
    if GPU:
        model.add(CuDNNLSTM(300, return_sequences=True))
    else:
        model.add(LSTM(300, return_sequences=True))
    
    # Layer 3
    if GPU:
        model.add(CuDNNLSTM(300))
    else:
        model.add(LSTM(300))

    # Layer 4
    model.add(Dense(512,activation='relu'))

    #model.add(Dropout(0.5))
    
    # Layer 5
    model.add(Dense(n_classes, activation='softmax'))

    if print_info:
        model.summary()
    
    return model
    
def Convolutional1DRecurrent(input_shape, n_classes, GPU=False, print_info = False):
    """5-layer neural network, with 1 1D convolutional and 2 LSTM and 2 fully connected layers."""

    model = Sequential()
  
    # Layer 1
    model.add(BatchNormalization(input_shape = input_shape))
    model.add(Conv1D(filters = 50, kernel_size = 11))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2))
        
    # Layer 2
    if GPU:
        model.add(CuDNNLSTM(300, return_sequences=True))
    else:
        model.add(LSTM(300, return_sequences=True))
    
    # Layer 3
    if GPU:
        model.add(CuDNNLSTM(300))
    else:
        model.add(LSTM(300))

    # Layer 4
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.3))
    
    #model.add(Dropout(0.5))
    
    # Layer 5
    model.add(Dense(n_classes, activation='softmax'))
    
    return model

def ConvolutionalDeepRecurrent(input_shape, n_classes, GPU=False, print_info = False):
    """7-layer neural network, with 3 1D convolutional, 2 LSTM and 2 fully connected layers."""

    model = Sequential()

    # Layer 1
    model.add(BatchNormalization(input_shape = input_shape))
    model.add(Conv1D(filters = 18, kernel_size=5, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    
    # Layer 2
    model.add(BatchNormalization())
    model.add(Conv1D(filters = 36, kernel_size=7, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    
    model.add(Dropout(0.2))
    
    # Layer 3
    model.add(BatchNormalization())
    model.add(Conv1D(filters = 72, kernel_size=7, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

    # Layer 4
    if GPU:
        model.add(CuDNNLSTM(60, return_sequences=True))
    else:
        model.add(LSTM(60, return_sequences=True))
    
    # Layer 5
    if GPU:
        model.add(CuDNNLSTM(60))
    else:
        model.add(LSTM(60))
   
    # Layer 6
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.3))

    # Layer 7
    model.add(Dense(n_classes, activation = 'softmax'))
    
    if print_info:
        model.summary()

    return model

########################################################################

# unused

def MotionDetection(input_shape, n_classes, print_info = False):
    
    model = Sequential()
  
    # Layer 0
    model.add(BatchNormalization(input_shape = input_shape))

    # Layer 1
    model.add(Conv1D(filters = 36,
                     kernel_size = 11,
                     strides=1))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2))
    
    # This layer dimension are automatically scanned in order to avoid updating by hand each time
    # model.add(Reshape((model.layers[2].output_shape[1], model.layers[2].output_shape[2] * model.layers[2].output_shape[3])))  

    # Layer 2
    model.add(LSTM(600, return_sequences=True))
    
    # Layer 3
    model.add(LSTM(600))
   
    # Layer 4
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.3))
    
    # Layer 5
    model.add(Dense(n_classes, activation = 'softmax'))
    
    if print_info:
        model.summary()

    return model

########################################################################

# models to be executed only on GPU, with TensorFlow backend
# some models have GRU units instead of LSTM

def Convolutional1DRecurrent_GPU(input_shape, n_classes, print_info = False):
    """5-layer neural network, with 1 1D convolutional and 2 LSTM and 2 fully connected layers."""

    model = Sequential()
  
    # Layer 1
    model.add(BatchNormalization(input_shape = input_shape))
    model.add(Conv1D(filters = 50, kernel_size = 11))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2))
        
    # Layer 2
    model.add(CuDNNGRU(300, return_sequences=True))
    
    # Layer 3
    model.add(CuDNNGRU(300))

    # Layer 4
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.3))
    
    #model.add(Dropout(0.5))
    
    # Layer 5
    model.add(Dense(n_classes, activation='softmax'))
    
    return model

def Convolutional2DRecurrent_GPU(input_shape, n_classes, print_info = False):
    """5-layer neural network, with 1 2D convolutional and 2 LSTM and 2 fully connected layers."""

    model = Sequential()
  
    # Layer 1
    model.add(BatchNormalization(input_shape = input_shape))
    model.add(Conv2D(filters = 50, kernel_size = (11,1), activation='relu'))
    #model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2,1)))
        
    # This layer dimension are automatically scanned in order to avoid updating by hand each time
    model.add(Reshape((model.layers[2].output_shape[1],model.layers[2].output_shape[2] * model.layers[2].output_shape[3])))  

    # Layer 2
    model.add(CuDNNLSTM(300, return_sequences=True))
    
    # Layer 3
    model.add(CuDNNLSTM(300))

    # Layer 4
    model.add(Dense(512,activation='relu'))

    #model.add(Dropout(0.5))
    
    # Layer 5
    model.add(Dense(n_classes, activation='softmax'))

    if print_info:
        model.summary()
    
    return model

def ConvolutionalDeepRecurrent_GPU(input_shape, n_classes, print_info = False):
    """7-layer neural network, with 3 1D convolutional, 2 LSTM and 2 fully connected layers."""

    model = Sequential()

    # Layer 1
    model.add(BatchNormalization(input_shape = input_shape))
    model.add(Conv1D(filters = 18, kernel_size=5, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    
    # Layer 2
    model.add(BatchNormalization())
    model.add(Conv1D(filters = 36, kernel_size=7, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    
    model.add(Dropout(0.2))
    
    # Layer 3
    model.add(BatchNormalization())
    model.add(Conv1D(filters = 72, kernel_size=7, strides=1, padding='same'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

    # Layer 4
    model.add(CuDNNLSTM(60, return_sequences=True))
    
    # Layer 5
    model.add(CuDNNLSTM(60))
   
    # Layer 6
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.3))

    # Layer 7
    model.add(Dense(n_classes, activation = 'softmax'))
    
    if print_info:
        model.summary()

    return model

