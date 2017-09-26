import numpy as np
from keras import callbacks
from keras import utils
from keras.models import *
from keras.layers import Input, merge, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, UpSampling2D, Dropout, Cropping2D, Activation
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
import os

NUM_PARTS = 4
NUM_POINTS = 2048
CATEGORY_NAME = 'Airplane'
CATEGORY_ID = '02691156'
X_TRAIN_PATH = '../data/prepared_old_train/' + CATEGORY_NAME + '_' + CATEGORY_ID + '_X_train.npy'
Y_TRAIN_PATH = '../data/prepared_old_train/' + CATEGORY_NAME + '_' + CATEGORY_ID + '_y_train.npy'

X_VAL_PATH = './data/' + CATEGORY_NAME + '_' + CATEGORY_ID + '_X_val.npy'
Y_VAL_PATH = './data/' + CATEGORY_NAME + '_' + CATEGORY_ID + '_y_val.npy'
IND_MAP_VAL_PATH = './data/' + CATEGORY_NAME + '_' + CATEGORY_ID + '_ind_map_val.npy'
LABEL_VAL_PATH = './data/val_label/' + CATEGORY_ID + '/*'

def custom_loss(y_true, y_pred):
    loss_iou=0
    # for i in range(y_true.shape[0]):
    # for i in range(32):
    y_true =K.reshape(y_true, (-1,2048,NUM_PARTS))
    y_pred =K.reshape(y_pred, (-1,2048,NUM_PARTS))
    for j in range(NUM_PARTS):
		dot = tf.multiply(y_true[:,j],y_pred[:,j])
		loss_iou += K.sum(K.flatten(dot))/K.sum(K.flatten(y_true[:,j]+y_pred[:,j]-dot))
    return 1-loss_iou/NUM_PARTS

class airplane_model():
    def __init__(self, n_pts = 2048):
        self.save_file = 'model_' + CATEGORY_NAME + '.hdf5'
        self.num_parts = NUM_PARTS
        self.n_pts = NUM_POINTS
        self.model = self.get_model()

    def get_conv(self, num_filters, layer_obj):
        return Conv2D(num_filters, (2,1), strides = (2,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(layer_obj)

    def get_up_conv_and_merge(self, num_filters, layer_obj, skip_con_layer):
        return merge([ skip_con_layer,Conv2DTranspose(num_filters,(2,1),strides = (2,1))(layer_obj) ],mode = 'concat', concat_axis = 3)

    def load_data(self):
        x_train = np.load(X_TRAIN_PATH)[:,:,:,1]
        x_train = x_train.reshape((-1,2048,3,1))
        print "x_train shape", x_train.shape
        y_train = np.load(Y_TRAIN_PATH)
        yt_shape = y_train.shape
        print "y_train shape", y_train.shape
        y_train = utils.to_categorical(y_train - 1,self.num_parts)
        y_train = np.reshape(y_train,(yt_shape[0],yt_shape[1],1,self.num_parts))
        print "y_train shape", y_train.shape

        x_val = np.load(X_VAL_PATH)[:,:,:,1]
        x_val = x_val.reshape((-1,2048,3,1))
        y_val = np.load(Y_VAL_PATH)
        yv_shape = y_val.shape
        y_val = utils.to_categorical(y_val - 1,self.num_parts)
        y_val = np.reshape(y_val,(yv_shape[0],yv_shape[1],1,self.num_parts))
        return x_train, y_train, x_val, y_val

    def get_model(self):
        inputs = Input((self.n_pts, 3,1))

        level1_conv = Conv2D(64, (2,3), strides = (2,1), activation = 'relu', padding = 'valid', kernel_initializer = 'glorot_normal')(inputs)
        level2_conv = self.get_conv(128,level1_conv)
        level3_conv = self.get_conv(256,level2_conv)
        level4_conv = self.get_conv(512,level3_conv)
        level5_conv = self.get_conv(1024,level4_conv)

        level4_up_conv = self.get_up_conv_and_merge(512,level5_conv,level4_conv)
        level3_up_conv = self.get_up_conv_and_merge(256,level4_up_conv,level3_conv)
        level2_up_conv = self.get_up_conv_and_merge(128,level3_up_conv,level2_conv)
        level1_up_conv = self.get_up_conv_and_merge(64,level2_up_conv,level1_conv)

        level0_out = Conv2DTranspose(self.num_parts,(2,1), strides = (2,1))(level1_up_conv)

        outputs = Lambda(self.softmax_,output_shape=(2048,1,self.num_parts))(level0_out)

        model = Model(input = inputs, output = outputs)
        model.compile(optimizer = Adam(lr = 1e-4, decay = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
        model.summary()
        return model

    def softmax_(self,x):
        return softmax(x,axis=3)

    def train(self):

        print("loading data")
        x_train, y_train, x_val, y_val = self.load_data()
        print("loading data done")
        # model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
        print('Fitting model...')

        mcb = My_Callback(x_val,y_val)
        prev_val_acc = 0
        if os.path.exists(self.save_file):
            self.model.load_weights(self.save_file)
            print("got weights")
        # model.fit(x_train, y_train, batch_size=16, epochs=5, verbose=1, shuffle=True, callbacks=[model_checkpoint])
        self.model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1, shuffle=True, callbacks = [mcb])
        print('Saving model..')
        self.model.save(self.save_file)


			# np.save('imgs_mask_test.npy', imgs_mask_test)




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


class My_Callback(callbacks.Callback):
    def __init__(self,x_val, y_val):
        self.X_val = x_val
        self.Y_val = y_val
        self.num_epochs = 0
        self.calc_epoch = 5

    def on_epoch_end(self, epoch, logs={}):
        if self.num_epochs%self.calc_epoch == 0:
            print('predict test data')
            val_score = self.model.evaluate(self.X_val,self.Y_val, batch_size=52, verbose=1)
            print val_score

            # P = self.model.predict(self.X_val, verbose = 0)
            # P = P.reshape((-1,2048,NUM_PARTS))
            # # print(P.shape)
            # indices = np.load(IND_MAP_VAL_PATH)
            # count = 0
            #
            # flists = sorted(glob.glob(LABEL_VAL_PATH))
            # IoU_sum = 0
            # for val_file in flists:
            # 	# print(val_file)
            # 	with open(val_file,'r') as myfile:
            # 		gt = np.loadtxt(myfile.readlines())
            # 	num_pts = len(gt)
            # 	seg_data = np.zeros((num_pts,NUM_PARTS))
            # 	num_exs = 1
            # 	if num_pts>2048:
            # 		num_exs = 2
            # 	for i in range(num_exs):
            # 		ind = indices[count]
            # 		prediction = P[count]
            # 		for j in range(2048):
            # 			seg_data[ind[j]] += prediction[j]
            # 		count += 1
            #
            # 	seg_pred = np.argmax(seg_data,axis=1) + 1
            # 	IoU_sum = IoU_sum +  IoU(gt,seg_pred)
            # 	# print('IIIIOOOOOOUUUUU: ' + str(IoU(gt,seg_pred)))
            # print('Mean IoU on val_data: ' + str(IoU_sum/len(flists)))
        self.num_epochs += 1.

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
	mymodel = airplane_model()
	# trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
	# non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
	#
	# print('Total params: {:,}'.format(trainable_count + non_trainable_count))
	# print('Trainable params: {:,}'.format(trainable_count))
	# print('Non-trainable params: {:,}'.format(non_trainable_count))
	mymodel.train()



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
