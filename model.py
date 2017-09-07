import numpy as np
from keras import utils
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Dropout, Cropping2D, Activation
from keras.layers.core import Reshape, Lambda
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.metrics import categorical_accuracy
from keras.activations import softmax
from keras import backend as K


class myUnet(object):

	def __init__(self, n_pts = 2048):

		self.n_pts = n_pts

	def load_data(self):
		x_data = './data/prepared/Motorbike_03790512_X_train.npy'
		y_data = './data/prepared/Motorbike_03790512_y_train.npy'
		x_train = np.load(x_data)
		print "x_train shape", x_train.shape
		y_train = np.load(y_data)
		yt_shape = y_train.shape
		print "y_train shape", y_train.shape
		y_train = utils.to_categorical(y_train - 1,6)
		y_train = np.reshape(y_train,(yt_shape[0],yt_shape[1],6))
		print "y_train shape", y_train.shape

		x_val = './data/prepared/Motorbike_03790512_X_val.npy'
		y_val = './data/prepared/Motorbike_03790512_y_val.npy'
		x_val = np.load(x_val)
		y_val = np.load(y_val)
		yv_shape = y_val.shape
		y_val = utils.to_categorical(y_val - 1,6)
		y_val = np.reshape(y_val,(yv_shape[0],yv_shape[1],6))
		return x_train, y_train, x_val, y_val

	def get_unet(self):


		inputs = Input((self.n_pts, 3,2))
		up_crop = Cropping2D(cropping=((0,1858),(0,0)))(inputs)
		up_shape = up_crop.shape
		up_crop = Lambda(lambda x: K.reverse(x,axes=1),output_shape=(190,3,2))(up_crop)
		print "up_crop shape:",up_crop.shape
		down_crop = Cropping2D(cropping=((1858,0),(0,0)))(inputs)
		down_shape = down_crop.shape
		down_crop = Lambda(lambda x: K.reverse(x,axes=1),output_shape=(190,3,2))(down_crop)
		print "down_crop shape:",down_crop.shape
		inputs_mirrored = merge([inputs,down_crop], mode = 'concat', concat_axis = 1)
		print "inputs shape:",inputs_mirrored.shape
		inputs_mirrored = merge([up_crop,inputs_mirrored], mode = 'concat', concat_axis = 1)
		print "inputs shape:",inputs_mirrored.shape

		conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(inputs_mirrored)
		print "conv1 shape:",conv1.shape
		conv1 = Conv2D(64, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(conv1)
		print "conv1 shape:",conv1.shape
		crop1 = Cropping2D(cropping=((184,184),(0,0)))(conv1)
		print "crop1 shape:",crop1.shape
		pool1 = AveragePooling2D(pool_size=(2, 1))(conv1)
		print "pool1 shape:",pool1.shape

		conv2 = Conv2D(128, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(pool1)
		print "conv2 shape:",conv2.shape
		conv2 = Conv2D(128, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(conv2)
		print "conv2 shape:",conv2.shape
		crop2 = Cropping2D(cropping=((88,88),(0,0)))(conv2)
		print "crop2 shape:",crop2.shape
		pool2 = MaxPooling2D(pool_size=(2,1))(conv2)
		print "pool2 shape:",pool2.shape

		conv3 = Conv2D(256, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(pool2)
		print "conv3 shape:",conv3.shape
		conv3 = Conv2D(256, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(conv3)
		print "conv3 shape:",conv3.shape
		crop3 = Cropping2D(cropping=((40,40),(0,0)))(conv3)
		print "crop3 shape:",crop3.shape
		pool3 = MaxPooling2D(pool_size=(2,1))(conv3)
		print "pool3 shape:",pool3.shape

		conv4 = Conv2D(512, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(pool3)
		print "conv4 shape:",conv4.shape
		conv4 = Conv2D(512, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(conv4)
		print "conv4 shape:",conv4.shape
		drop4 = Dropout(0.5)(conv4)
		crop4 = Cropping2D(cropping=((16,16),(0,0)))(drop4)
		print "crop4 shape:",crop4.shape
		pool4 = MaxPooling2D(pool_size=(2,1))(drop4)
		print "pool4 shape:",pool4.shape

		conv5 = Conv2D(1024, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(pool4)
		print "conv5 shape:",conv5.shape
		conv5 = Conv2D(1024, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(conv5)
		print "conv5 shape:",conv5.shape
		drop5 = Dropout(0.5)(conv5)
		crop5 = Cropping2D(cropping=((4,4),(0,0)))(drop5)
		print "crop5 shape:",crop5.shape
		pool5 = MaxPooling2D(pool_size=(2,1))(drop5)
		print "pool5 shape:",pool5.shape


		conv6 = Conv2D(2048, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(pool5)
		print "conv6 kerasshape:",conv6.shape
		conv6 = Conv2D(2048, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(conv6)
		print "conv6 shape:",conv6.shape
		# conv6 = Conv2D(2048, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(conv6)
		# print "conv6 shape:",conv6.shape
		drop6 = Dropout(0.5)(conv6)


		up7 = Conv2D(1024, (1,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(UpSampling2D(size = (2,1))(drop6))
		print "up7 shape:",up7.shape
		merge7 = merge([crop5,up7], mode = 'concat', concat_axis = 3)
		print "merge7 shape:",merge7.shape
		conv7 = Conv2D(1024, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(merge7)
		print "conv7 shape:",conv7.shape
		conv7 = Conv2D(1024, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(conv7)
		print "conv7 shape:",conv7.shape

		up8 = Conv2D(512, (1,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(UpSampling2D(size = (2,1))(conv7))
		print "up8 shape:",up8.shape
		merge8 = merge([crop4,up8], mode = 'concat', concat_axis = 3)
		print "merge8 shape:",merge8.shape
		conv8 = Conv2D(512, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(merge8)
		print "conv8 shape:",conv8.shape
		conv8 = Conv2D(512, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(conv8)
		print "conv8 shape:",conv8.shape

		up9 = Conv2D(256, (1,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(UpSampling2D(size = (2,1))(conv8))
		print "up9 shape:",up9.shape
		merge9 = merge([crop3,up9], mode = 'concat', concat_axis = 3)
		print "merge9 shape:",merge9.shape
		conv9 = Conv2D(256, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(merge9)
		print "merge9 shape:",merge9.shape
		conv9 = Conv2D(256, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(conv9)
		print "merge9 shape:",merge9.shape

		up10 = Conv2D(128, (1,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(UpSampling2D(size = (2,1))(conv9))
		print "up10 shape:",up10.shape
		merge10 = merge([crop2,up10], mode = 'concat', concat_axis = 3)
		print "merge10 shape:",merge10.shape
		conv10 = Conv2D(128, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(merge10)
		print "conv10 shape:",conv10.shape
		conv10 = Conv2D(128, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(conv10)
		print "conv10 shape:",conv10.shape

		up11 = Conv2D(64, (1,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(UpSampling2D(size = (2,1))(conv10))
		print "up11 shape:",up11.shape
		merge11 = merge([crop1,up11], mode = 'concat', concat_axis = 3)
		print "merge11 shape:",merge11.shape
		conv11 = Conv2D(64, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(merge11)
		print "conv11 shape:",conv11.shape
		conv11 = Conv2D(32,(3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(conv11)
		print "conv11 shape:",conv11.shape
		conv11 = Conv2D(16, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(conv11)
		print "conv11 shape:",conv11.shape
		conv11 = Conv2D(6, (3,1), padding = 'valid', kernel_initializer = 'glorot_normal')(conv11)
		print "conv11 shape:",conv11.shape
		conv11 = Reshape((2048, 6))(conv11)
		print "conv11 shape:",conv11.shape
		conv11 = Lambda(self.softmax_,output_shape=(2048,6))(conv11)
		print "conv11 shape:",conv11.shape
		# conv11 = up_crop = Lambda(lambda x: K.argmax(x,axis=2),output_shape=(2048,1))(conv11)

		model = Model(input = inputs, output = conv11)

		model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])

		return model

	def softmax_(self,x):
		return softmax(x,axis=2)

	def train(self):

		print("loading data")
		x_train, y_train, x_val, y_val = self.load_data()
		print("loading data done")
		model = self.get_unet()
		# model.load_weights('unet.hdf5')
		print("got unet")

		model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit(x_train, y_train, batch_size=2, epochs=200, verbose=1, shuffle=True, callbacks=[model_checkpoint])

		print('predict test data')
		val_score = model.evaluate(x_val,y_val, batch_size=52, verbose=1)

		print val_score
		# np.save('imgs_mask_test.npy', imgs_mask_test)


if __name__ == '__main__':
	myunet = myUnet()
	myunet.train()
