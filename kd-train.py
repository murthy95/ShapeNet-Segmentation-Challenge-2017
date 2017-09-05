import numpy as np
import keras
from keras.models import Sequential
from keras.layers import merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D

# loading data
X_train = np.load("")
X_val = np.load()
y_train = np.load()
y_val = np.load()

BATCH_SIZE = 111
NUM_CLASSES = 6
EPOCHS = 100

X_train = X_train.reshape((len(X_train),1024,3))
X_test = X_test.reshape((len(X_test),1024,3))
y_train = keras.utils.to_categorical(labels_train,NUM_CLASSES)
y_test = keras.utils.to_categorical(lables_test,NUM_CLASSES)

model = Sequential()
