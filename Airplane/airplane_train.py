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
NUM_POINTS = 2996
CATEGORY_NAME = 'Airplane'
CATEGORY_ID = '02691156'
X_TRAIN_PATH = './data/' + CATEGORY_NAME + '_' + CATEGORY_ID + '_X_train.npy'
Y_TRAIN_PATH = './data/' + CATEGORY_NAME + '_' + CATEGORY_ID + '_y_train.npy'

X_VAL_PATH = './data/' + CATEGORY_NAME + '_' + CATEGORY_ID + '_X_val.npy'
Y_VAL_PATH = './data/' + CATEGORY_NAME + '_' + CATEGORY_ID + '_y_val.npy'
IND_MAP_VAL_PATH = './data/' + CATEGORY_NAME + '_' + CATEGORY_ID + '_ind_map_val.npy'
LABEL_VAL_PATH = './data/val_label/' + CATEGORY_ID + '/*'

class airplane_model():
    def __init__(self, n_pts = 2048):
        self.save_file = 'model_' + CATEGORY_NAME + '.hdf5'
        self.num_parts = NUM_PARTS
        self.n_pts = NUM_POINTS
        self.model = self.get_model()

    def get_conv(self, num_filters, layer_obj):
        conv1 = Conv2D(num_filters, (7,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(layer_obj)
        conv1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv1)
        return Activation('relu')(conv1)

    def get_up_conv(self, num_filters, layer_obj):
        conv1 = Conv2DTranspose(num_filters, (7,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(layer_obj)
        conv1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv1)
        return Activation('relu')(conv1)

    def get_up_conv_and_merge(self, num_filters, filter_size, layer_obj, skip_con_layer):
        half2 = Conv2DTranspose(num_filters, (2,1), strides = (2,1))(layer_obj)
        half1 = Conv2D(num_filters, filter_size, activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(skip_con_layer)
        merged = merge([half1,half2], mode = 'concat', concat_axis = 3)
        conv1 = Conv2DTranspose(num_filters, (7,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(merged)
        conv1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv1)
        return Activation('relu')(conv1)

    def load_data(self):
        x_train = np.load(X_TRAIN_PATH)
        x_train = x_train.reshape((-1,NUM_POINTS,4,1))
        print "x_train shape", x_train.shape
        y_train = np.load(Y_TRAIN_PATH)
        yt_shape = y_train.shape
        print "y_train shape", y_train.shape
        y_train = utils.to_categorical(y_train - 1,self.num_parts)
        y_train = np.reshape(y_train,(yt_shape[0],yt_shape[1],1,self.num_parts))
        print "y_train shape", y_train.shape

        x_val = np.load(X_VAL_PATH)
        x_val = x_val.reshape((-1,NUM_POINTS,4,1))
        y_val = np.load(Y_VAL_PATH)
        yv_shape = y_val.shape
        y_val = utils.to_categorical(y_val - 1,self.num_parts)
        y_val = np.reshape(y_val,(yv_shape[0],yv_shape[1],1,self.num_parts))
        return x_train, y_train, x_val, y_val

    def get_model(self):
        inputs = Input((self.n_pts, 4,1))

        level1a_conv = Conv2D(32, (7,4), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(inputs)
        level1a_conv = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(level1a_conv)
        level1b_conv = self.get_conv(32, Activation('relu')(level1a_conv) )

        level2a_conv = self.get_conv(64, AveragePooling2D(pool_size=(2, 1))(level1b_conv) )
        level2b_conv = self.get_conv(64, level2a_conv)

        level3a_conv = self.get_conv(128, MaxPooling2D(pool_size=(2,1))(level2b_conv) )
        level3b_conv = self.get_conv(128, level3a_conv)

        level4a_conv = self.get_conv(256, MaxPooling2D(pool_size=(2,1))(level3b_conv) )
        level4b_conv = self.get_conv(256, level4a_conv)

        level5a_conv = self.get_conv(512, MaxPooling2D(pool_size=(2,1))(Dropout(0.5)(level4b_conv)) )
        level5b_conv = self.get_conv(512, level5a_conv)

        level6a_conv = self.get_conv(1024, MaxPooling2D(pool_size=(2,1))(Dropout(0.5)(level5b_conv)) )
        level6b_conv = Dropout(0.5)(self.get_conv(1024, level6a_conv))

        level5a_up_conv = self.get_up_conv_and_merge(512,(25,1),level6b_conv,level5b_conv)
        level5b_up_conv = self.get_up_conv(512, level5a_up_conv)

        level4a_up_conv = self.get_up_conv_and_merge(256,(49,1),level5b_up_conv,level4b_conv)
        level4b_up_conv = self.get_up_conv(256, level4a_up_conv)

        level3a_up_conv = self.get_up_conv_and_merge(128,(97,1),level4b_up_conv,level3b_conv)
        level3b_up_conv = self.get_up_conv(128, level3a_up_conv)

        level2a_up_conv = self.get_up_conv_and_merge(64,(193,1),level3b_up_conv,level2b_conv)
        level2b_up_conv = self.get_up_conv(64, level2a_up_conv)

        level1a_up_conv_h2 = Conv2DTranspose(32, (2,1), strides = (2,1))(level2b_up_conv)
        level1a_up_conv_h1 = Conv2D(32, (385,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(level1b_conv)
        level1a_up_conv_merged = merge([level1a_up_conv_h1,level1a_up_conv_h2], mode = 'concat', concat_axis = 3)
        level1a_up_conv = Conv2DTranspose(32, (409,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(level1a_up_conv_merged)
        level1a_up_conv = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(level1a_up_conv)
        level1a_up_conv = Activation('relu')(level1a_up_conv)

        level1_out = self.get_conv(32,level1a_up_conv)
        level1_out = Conv2D(NUM_PARTS, (7,1), activation = 'linear', padding = 'valid', kernel_initializer = 'glorot_normal')(level1_out)

        outputs = Lambda(self.softmax_,output_shape=(NUM_POINTS,1,self.num_parts))(level1_out)

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
        print('Fitting model...')

        mcb = My_Callback(x_val,y_val)
        prev_val_acc = 0
        # if os.path.exists(self.save_file):
        #     self.model.load_weights(self.save_file)
        #     print("got weights")

        self.model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1, shuffle=True, callbacks = [mcb])
        # self.model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1, shuffle=True)
        print('Saving model..')
        self.model.save(self.save_file)

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
        self.calc_epoch = 1

    def on_epoch_end(self, epoch, logs={}):
        if self.num_epochs%self.calc_epoch == 0:
            print('predict test data')
            val_score = self.model.evaluate(self.X_val,self.Y_val, batch_size=52, verbose=1)
            print val_score

            # P = self.model.predict(self.X_val, verbose = 0)
            # indices = np.load(IND_MAP_VAL_PATH)
            # count = 0
            #
            # flists = sorted(glob.glob(LABEL_VAL_PATH))
            # IoU_sum = 0
            # Acc_sum = 0
            # for val_file in flists:
            # 	# print(val_file)
            # 	with open(val_file,'r') as myfile:
            # 		gt = np.loadtxt(myfile.readlines())
            # 	num_pts = len(gt)
            # 	seg_data = np.zeros((num_pts,1,NUM_PARTS))
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
            # 	m_iou, m_Acc = IoU(gt,seg_pred)
            # 	IoU_sum = IoU_sum + m_iou
            # 	Acc_sum += m_Acc
            # 	# print('IIIIOOOOOOUUUUU: ' + str(IoU(gt,seg_pred)))
            # print('Mean IoU on val_data: ' + str(IoU_sum/len(flists)))
            # print('Mean acc. on val_data: ' + str(Acc_sum/len(flists)))
        self.num_epochs += 1.

# def prepare_seg_path(original_path):
# 	path_segs = original_path.split('.')
# 	path_segs = path_segs[1].split('/')
# 	return './temp_segs/' + path_segs[len(path_segs)-1] + '.seg'

def IoU(gt_seg,pred_seg):
	tp, tn, fp, fn = np.zeros(NUM_PARTS), np.zeros(NUM_PARTS), np.zeros(NUM_PARTS),  np.zeros(NUM_PARTS)
	for i in range(NUM_PARTS):
		pred_true_inds = np.where(pred_seg == (i+1))[0]
		pred_false_inds = np.where(pred_seg != (i+1))[0]
		# print(gt_seg[pred_true_inds])
		tp[i] = len(np.where(gt_seg[pred_true_inds] == (i+1) )[0])
		tn[i] = len(np.where(gt_seg[pred_false_inds] != (i+1) )[0])
		fp[i] = len(pred_true_inds) - tp[i]
		fn[i] = len(np.where(gt_seg[pred_false_inds] == (i+1) )[0])
	# print(tp)
	# print(fp)
	# print(fn)
	denom = (tp + fp + fn)
	iou = tp / denom

	# avoiding division by zero
	iou[np.where(denom == 0)[0]]  = 0
	# print(iou)
	return sum(iou)/NUM_PARTS, sum((tp + tn)/(tp + tn + fp + fn))/NUM_PARTS


if __name__ == '__main__':
	mymodel = airplane_model()
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
