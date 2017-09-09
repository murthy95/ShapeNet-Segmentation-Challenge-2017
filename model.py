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
from keras.layers.normalization import BatchNormalization
import glob
import tensorflow as tf

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

		conv1 = Conv2D(64, (3,3), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(inputs_mirrored)
		print "conv1 shape:",conv1.shape
		conv1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv1)
		conv1 = Activation('relu')(conv1)
		conv1 = Conv2D(64, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(conv1)
		print "conv1 shape:",conv1.shape
		conv1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv1)
		conv1 = Activation('relu')(conv1)
		crop1 = Cropping2D(cropping=((184,184),(0,0)))(conv1)
		print "crop1 shape:",crop1.shape
		pool1 = AveragePooling2D(pool_size=(2, 1))(conv1)
		print "pool1 shape:",pool1.shape

		conv2 = Conv2D(128, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(pool1)
		print "conv2 shape:",conv2.shape
		conv2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv2)
		conv2 = Activation('relu')(conv2)
		conv2 = Conv2D(128, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(conv2)
		print "conv2 shape:",conv2.shape
		conv2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv2)
		conv2 = Activation('relu')(conv2)
		crop2 = Cropping2D(cropping=((88,88),(0,0)))(conv2)
		print "crop2 shape:",crop2.shape
		pool2 = MaxPooling2D(pool_size=(2,1))(conv2)
		print "pool2 shape:",pool2.shape

		conv3 = Conv2D(256, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(pool2)
		print "conv3 shape:",conv3.shape
		conv3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv3)
		conv3 = Activation('relu')(conv3)
		conv3 = Conv2D(256, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(conv3)
		print "conv3 shape:",conv3.shape
		conv3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv3)
		conv3 = Activation('relu')(conv3)
		crop3 = Cropping2D(cropping=((40,40),(0,0)))(conv3)
		print "crop3 shape:",crop3.shape
		pool3 = MaxPooling2D(pool_size=(2,1))(conv3)
		print "pool3 shape:",pool3.shape

		conv4 = Conv2D(512, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(pool3)
		print "conv4 shape:",conv4.shape
		conv4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv4)
		conv4 = Activation('relu')(conv4)
		conv4 = Conv2D(512, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(conv4)
		print "conv4 shape:",conv4.shape
		conv4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv4)
		conv4 = Activation('relu')(conv4)
		drop4 = Dropout(0.5)(conv4)
		crop4 = Cropping2D(cropping=((16,16),(0,0)))(drop4)
		print "crop4 shape:",crop4.shape
		pool4 = MaxPooling2D(pool_size=(2,1))(drop4)
		print "pool4 shape:",pool4.shape

		conv5 = Conv2D(1024, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(pool4)
		print "conv5 shape:",conv5.shape
		conv5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv5)
		conv5 = Activation('relu')(conv5)
		conv5 = Conv2D(1024, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(conv5)
		print "conv5 shape:",conv5.shape
		conv5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv5)
		conv5 = Activation('relu')(conv5)
		drop5 = Dropout(0.5)(conv5)
		crop5 = Cropping2D(cropping=((4,4),(0,0)))(drop5)
		print "crop5 shape:",crop5.shape
		pool5 = MaxPooling2D(pool_size=(2,1))(drop5)
		print "pool5 shape:",pool5.shape


		conv6 = Conv2D(2048, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(pool5)
		print "conv6 kerasshape:",conv6.shape
		conv6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv6)
		conv6 = Activation('relu')(conv6)
		conv6 = Conv2D(2048, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(conv6)
		print "conv6 shape:",conv6.shape
		conv6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv6)
		conv6 = Activation('relu')(conv6)
		# conv6 = Conv2D(2048, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(conv6)
		# print "conv6 shape:",conv6.shape
		drop6 = Dropout(0.5)(conv6)


		up7 = Conv2D(1024, (1,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(UpSampling2D(size = (2,1))(drop6))
		print "up7 shape:",up7.shape
		merge7 = merge([crop5,up7], mode = 'concat', concat_axis = 3)
		print "merge7 shape:",merge7.shape
		conv7 = Conv2D(1024, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(merge7)
		print "conv7 shape:",conv7.shape
		conv7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv7)
		conv7 = Activation('relu')(conv7)
		conv7 = Conv2D(1024, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(conv7)
		print "conv7 shape:",conv7.shape
		conv7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv7)
		conv7 = Activation('relu')(conv7)

		up8 = Conv2D(512, (1,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(UpSampling2D(size = (2,1))(conv7))
		print "up8 shape:",up8.shape
		merge8 = merge([crop4,up8], mode = 'concat', concat_axis = 3)
		print "merge8 shape:",merge8.shape
		conv8 = Conv2D(512, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(merge8)
		print "conv8 shape:",conv8.shape
		conv8 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv8)
		conv8 = Activation('relu')(conv8)
		conv8 = Conv2D(512, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(conv8)
		print "conv8 shape:",conv8.shape
		conv8 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv8)
		conv8 = Activation('relu')(conv8)

		up9 = Conv2D(256, (1,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(UpSampling2D(size = (2,1))(conv8))
		print "up9 shape:",up9.shape
		merge9 = merge([crop3,up9], mode = 'concat', concat_axis = 3)
		print "merge9 shape:",merge9.shape
		conv9 = Conv2D(256, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(merge9)
		print "merge9 shape:",merge9.shape
		conv9 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv9)
		conv9 = Activation('relu')(conv9)
		conv9 = Conv2D(256, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(conv9)
		print "merge9 shape:",merge9.shape
		conv9 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv9)
		conv9 = Activation('relu')(conv9)

		up10 = Conv2D(128, (1,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(UpSampling2D(size = (2,1))(conv9))
		print "up10 shape:",up10.shape
		merge10 = merge([crop2,up10], mode = 'concat', concat_axis = 3)
		print "merge10 shape:",merge10.shape
		conv10 = Conv2D(128, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(merge10)
		print "conv10 shape:",conv10.shape
		conv10 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv10)
		conv10 = Activation('relu')(conv10)
		conv10 = Conv2D(128, (3,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(conv10)
		print "conv10 shape:",conv10.shape
		conv10 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv10)
		conv10 = Activation('relu')(conv10)

		up11 = Conv2D(64, (1,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(UpSampling2D(size = (2,1))(conv10))
		print "up11 shape:",up11.shape
		merge11 = merge([crop1,up11], mode = 'concat', concat_axis = 3)
		print "merge11 shape:",merge11.shape
		conv11 = Conv2D(64, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(merge11)
		print "conv11 shape:",conv11.shape
		conv11 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv11)
		conv11 = Activation('relu')(conv11)
		conv11 = Conv2D(32,(3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(conv11)
		print "conv11 shape:",conv11.shape
		conv11 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv11)
		conv11 = Activation('relu')(conv11)
		conv11 = Conv2D(16, (3,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(conv11)
		print "conv11 shape:",conv11.shape
		conv11 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv11)
		conv11 = Activation('relu')(conv11)
		conv11 = Conv2D(6, (3,1), padding = 'valid', kernel_initializer = 'glorot_normal')(conv11)
		conv11 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv11)
		print "conv11 shape:",conv11.shape
		conv11 = Reshape((2048, 6))(conv11)
		print "conv11 shape:",conv11.shape
		conv11 = Lambda(self.softmax_,output_shape=(2048,6))(conv11)
		print "conv11 shape:",conv11.shape
		# conv11 = up_crop = Lambda(lambda x: K.argmax(x,axis=2),output_shape=(2048,1))(conv11)

		model = Model(input = inputs, output = conv11)

		model.compile(optimizer = Adam(lr = 1e-4, decay = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

		return model

	def softmax_(self,x):
		return softmax(x,axis=2)

	def train(self):

		print("loading data")
		x_train, y_train, x_val, y_val = self.load_data()
		print("loading data done")
		model = self.get_unet()


		model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')

		for j in range(100):
			model.load_weights('unet.hdf5')
			print("got unet")
			model.fit(x_train, y_train, batch_size=16, epochs=1, verbose=1, validation_data = (x_val,y_val), shuffle=True, callbacks=[model_checkpoint])

			# print('predict test data')
			# val_score = model.evaluate(x_val,y_val, batch_size=52, verbose=1)
			#
			# print val_score
			# np.save('imgs_mask_test.npy', imgs_mask_test)

			print('!!!!!!!!!!!!! Calculating mean IoU !!!!!!!!!!!!')
			P = model.predict(x_val, verbose = 0)
			indices = np.load('./data/prepared/Motorbike_03790512_ind_map_val.npy')
			count = 0

			flists = sorted(glob.glob('./data/val_label/03790512/*'))
			IoU_sum = 0
			for val_file in flists:
				# print(val_file)
				with open(val_file,'r') as myfile:
					gt = np.loadtxt(myfile.readlines())
				num_pts = len(gt)
				seg_data = np.zeros((num_pts,6))
				num_exs = 1
				if num_pts>2048:
					num_exs = 2
				for i in range(num_exs):
					ind = indices[count]
					prediction = P[count]
					for j in range(2048):
						seg_data[ind[j]] += prediction[j]
					count += 1

				seg_pred = np.argmax(seg_data,axis=1) + 1
				IoU_sum = IoU_sum +  IoU(gt,seg_pred)
				# print('IIIIOOOOOOUUUUU: ' + str(IoU(gt,seg_pred)))
			print('Mean IoU on val_data: ' + str(IoU_sum/len(flists)))


	# def predict(self):
	# 	print("loading data")
	# 	x_train, y_train, x_val, y_val = self.load_data()
	# 	print("loading data done")
	# 	model = self.get_unet()
	# 	print("got unet")
	# 	model.load_weights('unet.hdf5')
	# 	print("loaded weights")
	# 	predictions = model.predict(x_val, batch_size = 1)
	# 	return predictions


def prepare_seg_path(original_path):
	path_segs = original_path.split('.')
	path_segs = path_segs[1].split('/')
	return './temp_segs/' + path_segs[len(path_segs)-1] + '.seg'

def mean_IoU(y_true, y_pred):
	y_pred = K.argmax(y_pred,axis=2)
	y_true = K.argmax(y_true,axis=2)
	score, up_opt = tf.metrics.mean_iou(K.flatten(y_true), K.flatten(y_pred), 6)
	K.get_session().run(tf.local_variables_initializer())
	with tf.control_dependencies([up_opt]):
		score = tf.identity(score)
	return score

def IoU(gt_seg,pred_seg):
	tp, fp, fn = np.zeros(6), np.zeros(6), np.zeros(6)
	for i in range(6):
		pred_true_inds = np.where(pred_seg == (i+1))[0]
		pred_false_inds = np.where(pred_seg != (i+1))[0]
		# print(gt_seg[pred_true_inds])
		tp[i] = len(np.where(gt_seg[pred_true_inds] == (i+1) )[0])
		fp[i] = len(pred_true_inds) - tp[i]
		fn[i] = len(np.where(gt_seg[pred_false_inds] == (i+1) )[0])
	# print(tp)
	# print(fp)
	# print(fn)
	denom = (tp + fp + fn)
	iou = tp / denom
	iou[np.where(denom == 0)[0]]  = 0
	# print(iou)
	return sum(iou)/6


if __name__ == '__main__':
	myunet = myUnet()
	myunet.train()



		# P = myunet.predict()
		#
		# indices = np.load('./data/prepared/Motorbike_03790512_ind_map_val.npy')
		# count = 0
		#
		# flists = sorted(glob.glob('./data/val_data/03790512/*'))
		# for val_file in flists:
		# 	print(val_file)
		# 	with open(val_file,'r') as myfile:
		# 		num_pts = len(myfile.readlines())
		# 	seg_data = np.zeros((num_pts,6))
		# 	num_exs = 1
		# 	if num_pts>2048:
		# 		num_exs = 2
		# 	for i in range(num_exs):
		# 		ind = indices[count]
		# 		prediction = P[count]
		# 		for j in range(2048):
		# 			seg_data[ind[j]] += prediction[j]
		# 		count += 1
		# 	seg_file = prepare_seg_path(val_file)
		# 	np.savetxt(seg_file,np.argmax(seg_data,axis=1) + 1,fmt='%1.f')
