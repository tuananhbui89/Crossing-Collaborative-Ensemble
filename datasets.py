import keras
from keras.datasets import mnist, cifar10, cifar100, fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
		
class Dataset(object):

	def __init__(self, ds, num_models=1, 
				subtract_pixel_mean=True, 
				clip_min=0., 
				clip_max=1.):

		super(Dataset, self).__init__()

		self.clip_min = clip_min
		self.clip_max = clip_max

		if ds == 'mnist':
			(x_train, y_train), (x_test, y_test) = mnist.load_data()
			x_train = np.expand_dims(x_train, axis=3)
			x_test = np.expand_dims(x_test, axis=3)
			nb_class = 10 
		elif ds == 'cifar10': 
			(x_train, y_train), (x_test, y_test) = cifar10.load_data()
			nb_class = 10 
		elif ds == 'cifar100': 
			(x_train, y_train), (x_test, y_test) = cifar100.load_data()
			nb_class = 100
		elif ds == 'fashion': 
			(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
			x_train = np.expand_dims(x_train, axis=3)
			x_test = np.expand_dims(x_test, axis=3)
			nb_class = 10 			

		# Normalize data.
		self.x_train = x_train.astype('float32') / 255
		self.x_test = x_test.astype('float32') / 255

		self.y_train = keras.utils.to_categorical(y_train, nb_class)
		self.y_test = keras.utils.to_categorical(y_test, nb_class)

		# get infor 
		self.x_shape = x_train.shape[1:]


		# 
		if subtract_pixel_mean:
			# x_train_mean = np.mean(x_train, axis=0)
			self.x_train_mean = np.mean(self.x_train)
			self.x_train -= self.x_train_mean
			self.x_test -= self.x_train_mean
			self.clip_min -= self.x_train_mean
			self.clip_max -= self.x_train_mean
			self.clip_min = float(self.clip_min)
			self.clip_max = float(self.clip_max)

		else: 
			self.x_train_mean = 0.

		if num_models > 1: 
			"""
			if num_models > 1, need duplicate the true labels by #num_models times.
			for example:  y1 = model_1(x); y2 = model_2(x); y = [y1, y2]
			"""
			y_train_2 = []
			y_test_2 = []
			for _ in range(num_models):
				y_train_2.append(self.y_train)
				y_test_2.append(self.y_test)
			y_train_2 = np.concatenate(y_train_2, axis=-1)
			y_test_2 = np.concatenate(y_test_2, axis=-1)

			self.y_train = y_train_2
			self.y_test = y_test_2

		self.datagen = ImageDataGenerator(
			# set input mean to 0 over the dataset
			featurewise_center=False,
			# set each sample mean to 0
			samplewise_center=False,
			# divide inputs by std of dataset
			featurewise_std_normalization=False,
			# divide each input by its std
			samplewise_std_normalization=False,
			# apply ZCA whitening
			zca_whitening=False,
			# epsilon for ZCA whitening
			zca_epsilon=1e-06,
			# randomly rotate images in the range (deg 0 to 180)
			rotation_range=0,
			# randomly shift images horizontally
			width_shift_range=0.1,
			# randomly shift images vertically
			height_shift_range=0.1,
			# set range for random shear
			shear_range=0.,
			# set range for random zoom
			zoom_range=0.,
			# set range for random channel shifts
			channel_shift_range=0.,
			# set mode for filling points outside the input boundaries
			fill_mode='nearest',
			# value used for fill_mode = "constant"
			cval=0.,
			# randomly flip images
			horizontal_flip=True,
			# randomly flip images
			vertical_flip=False,
			# set rescaling factor (applied before any other transformation)
			rescale=None,
			# set function that will be applied on each input
			preprocessing_function=None,
			# image data format, either "channels_first" or "channels_last"
			data_format=None,
			# fraction of images reserved for validation (strictly between 0 and 1)
			validation_split=0.0)

		# Compute quantities required for featurewise normalization
		# (std, mean, and principal components if ZCA whitening is applied).
		self.datagen.fit(self.x_train)

		print('Finish data processing')

	def next_batch(self, bs): 
		return self.datagen.flow(self.x_train, self.y_train, batch_size=bs)


