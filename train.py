import keras
import keras.backend as K

'''
asssuming that the input received is a numpy array of training points and their
corresponding segmentation label
input nx1x3 tensor
output map nxk for k label point cloud segmentation
using fully covolutional architecture inspired by 'U-Net'
'''

from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.layers import Dense, Input, Softmax

def cross_entropy(y_true, y_predicted): #ytrue of size nx1 and y_predicted of size nx1x3
    y_predicted = np.argmax(y_predicted, axis=2)
        return categorical_crossentropy(y_true, y_predicted)

n =2048
input_ = Input(shape=(n,3,2))
a1 =
