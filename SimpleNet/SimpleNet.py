import keras

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D

from keras.initializers import glorot_normal, RandomNormal, Zeros
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

# SimpleNet
def SimpleNet(H,W,C,num_classes):
	"""
	Keras implementation of SimpleNet (arXiv: 1608.06037)
	(https://arxiv.org/abs/1608.06037)

	Inputs:
	- H: number of vertical pixels
	- W: number of horizontal pixels
	- C: number of input channels (generally number of colour channels)
	- num_classes: total number of final categories

	Returns:
	- model: SimpleNet model
	"""

	act = 'relu'
	model = Sequential()

	# Cell 1
	model.add(Conv2D(filters=64, kernel_size=3, strides=1, use_bias=True, padding='same', kernel_initializer=glorot_normal(), input_shape=[H,W,C], name='conv1'))
	model.add(BatchNormalization(momentum=0.95, center=True, scale=True, name='batchnorm1'))
	model.add(Activation(act, name='act1'))
	model.add(Dropout(0.2, name='dropout1'))

	# Cell 2
	model.add(Conv2D(filters=128, kernel_size=3, strides=1, use_bias=True, padding='same', kernel_initializer=glorot_normal(), name='conv2'))
	model.add(BatchNormalization(momentum=0.95, center=True, scale=True, name='batchnorm2'))
	model.add(Activation(act, name='act2'))
	model.add(Dropout(0.2, name='dropout2'))

	# Cell 3
	model.add(Conv2D(filters=128, kernel_size=3, strides=1, use_bias=True, padding='same', kernel_initializer=RandomNormal(stddev=0.01), name='conv3'))
	model.add(BatchNormalization(momentum=0.95, center=True, scale=True, name='batchnorm3'))
	model.add(Activation(act, name='act3'))
	model.add(Dropout(0.2, name='dropout3'))

	# Cell 4
	model.add(Conv2D(filters=128, kernel_size=3, strides=1, use_bias=True, padding='same', kernel_initializer=RandomNormal(stddev=0.01), name='conv4'))
	model.add(BatchNormalization(momentum=0.95, center=True, scale=True, name='batchnorm4'))
	model.add(Activation(act, name='act4'))
	# MaxPool 4
	model.add(MaxPooling2D(pool_size=2, strides=2, name='pool4'))
	model.add(Dropout(0.2, name='dropout4'))

	# Cell 5
	model.add(Conv2D(filters=128, kernel_size=3, strides=1, use_bias=True, padding='same', kernel_initializer=RandomNormal(stddev=0.01), name='conv5'))
	model.add(BatchNormalization(momentum=0.95, center=True, scale=True, name='batchnorm5'))
	model.add(Activation(act, name='act5'))
	model.add(Dropout(0.2, name='dropout5'))

	# Cell 6
	model.add(Conv2D(filters=128, kernel_size=3, strides=1, use_bias=True, padding='same', kernel_initializer=glorot_normal(), name='conv6'))
	model.add(BatchNormalization(momentum=0.95, center=True, scale=True, name='batchnorm6'))
	model.add(Activation(act, name='act6'))
	model.add(Dropout(0.2, name='dropout6'))

	# Cell 7
	model.add(Conv2D(filters=256, kernel_size=3, strides=1, use_bias=True, padding='same', kernel_initializer=glorot_normal(), name='conv7'))
	# MaxPool 7
	model.add(MaxPooling2D(pool_size=2, strides=2, name='pool7'))
	model.add(BatchNormalization(momentum=0.95, center=True, scale=True, name='batchnorm7'))
	model.add(Activation(act, name='act7'))
	model.add(Dropout(0.2, name='dropout7'))

	# Cell 8
	model.add(Conv2D(filters=256, kernel_size=3, strides=1, use_bias=True, padding='same', kernel_initializer=glorot_normal(), name='conv8'))
	model.add(BatchNormalization(momentum=0.95, center=True, scale=True, name='batchnorm8'))
	model.add(Activation(act, name='act8'))
	model.add(Dropout(0.2, name='dropout8'))

	# Cell 9
	model.add(Conv2D(filters=256, kernel_size=3, strides=1, use_bias=True, padding='same', kernel_initializer=glorot_normal(), name='conv9'))
	model.add(BatchNormalization(momentum=0.95, center=True, scale=True, name='batchnorm9'))
	model.add(Activation(act, name='act9'))
	model.add(Dropout(0.2, name='dropout9'))
	# MaxPool 9
	model.add(MaxPooling2D(pool_size=2, strides=2, name='pool9'))

	# Cell 10
	model.add(Conv2D(filters=512, kernel_size=3, strides=1, use_bias=True, padding='same', kernel_initializer=glorot_normal(), name='conv10'))
	model.add(BatchNormalization(momentum=0.95, center=True, scale=True, name='batchnorm10'))
	model.add(Activation(act, name='act10'))
	model.add(Dropout(0.2, name='dropout10'))

	# Cell 11
	model.add(Conv2D(filters=2048, kernel_size=1, strides=1, use_bias=True, padding='same', kernel_initializer=glorot_normal(), bias_initializer=Zeros(), name='conv11'))
	model.add(Activation(act, name='act11'))
	model.add(Dropout(0.2, name='dropout11'))

	# Cell 12
	model.add(Conv2D(filters=256, kernel_size=1, strides=1, use_bias=True, padding='same', kernel_initializer=glorot_normal(), bias_initializer=Zeros(), name='conv12'))
	model.add(Activation(act, name='act12'))
	# MaxPool 12
	model.add(MaxPooling2D(pool_size=2, strides=2, name='pool12'))
	model.add(Dropout(0.2, name='dropout12'))

	# Cell 13
	model.add(Conv2D(filters=256, kernel_size=3, strides=1, use_bias=True, padding='same', kernel_initializer=glorot_normal(), bias_initializer=Zeros(), name='conv13'))
	model.add(Activation(act, name='act13'))
	# MaxPool 13
	model.add(MaxPooling2D(pool_size=2, strides=2, name='pool13'))

	# Final Classifier
	model.add(Flatten(name='flatten'))
	model.add(Dense(num_classes, activation='softmax', name='softmax',
	kernel_initializer=glorot_normal(), bias_initializer=Zeros()))

	return model
