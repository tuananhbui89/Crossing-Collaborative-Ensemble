from __future__ import print_function

import os 
import sys 

import tensorflow
if tensorflow.__version__ == '2.0.0': 
	import tensorflow.compat.v1 as tf 
else: 
	import tensorflow as tf 

import numpy as np
import logging
from cleverhans.utils import batch_indices, _ArgsWrapper, create_logger

_logger = create_logger("cleverhans.utils.tf")
_logger.setLevel(logging.INFO)
np_dtype = np.dtype('float32')
tf_dtype = tf.as_dtype('float32')


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('lamda', 2.0, "lamda for Ensemble Entropy(EE)")
tf.app.flags.DEFINE_float('log_det_lamda', 0.5, "lamda for non-ME")
tf.app.flags.DEFINE_float('label_smooth', 1.0, "label smooth")


tf.app.flags.DEFINE_integer('epoch', 0, "epoch of the checkpoint to load")
tf.app.flags.DEFINE_string('dataset', 'cifar10', "mnist or cifar10 or cifar100")
tf.app.flags.DEFINE_string('defense', 'pie', "adv_en, adp or pie")
tf.app.flags.DEFINE_string('model', 'cnn', "resnet20, cnn or wideresnet")
tf.app.flags.DEFINE_integer('num_models', 2, "The num of models in the ensemble")
tf.app.flags.DEFINE_integer('batch_size', 64, "batch_size")
tf.app.flags.DEFINE_string('inf', 'none', "additional information")
tf.app.flags.DEFINE_integer('num_epochs', 180, "number training epoch")
tf.app.flags.DEFINE_bool('augmentation', False, "whether use data augmentation")
tf.app.flags.DEFINE_float('wdm', 0.0, "demoting parameter")
tf.app.flags.DEFINE_float('wpm', 0.0, "crossing promoting parameter")
tf.app.flags.DEFINE_float('wce', 1.0, "direct promoting parameter")
tf.app.flags.DEFINE_string('setting', 'mine', "adp setting or my setting")
print(vars(FLAGS))

zero = tf.constant(0, dtype=tf.float32)
# Training parameters
if FLAGS.dataset=='cifar100':
    num_classes = 100
elif FLAGS.dataset=='cifar10' or FLAGS.dataset=='mnist':
    num_classes = 10
log_offset = 1e-20
det_offset = 1e-6


# Assign setting for adversarial training 

if FLAGS.setting == 'adp': 
	assert(FLAGS.augmentation)
	assert(FLAGS.batch_size==64)
	ARGS = dict()

	if FLAGS.dataset=='cifar10':
		ARGS['eps'] = tf.random_uniform((), 0.01, 0.05)
		ARGS['eta'] = ARGS['eps'] / 10. 
		ARGS['def_iter'] = 10 
		ARGS['attack_method'] = 'MadryEtAl'

	setup = [
		('ds={}',		FLAGS.dataset), 
		('defense={}',	FLAGS.defense), 
		('model={}',	FLAGS.model), 
		('bs={}', 		FLAGS.batch_size),
		('setting={}', 	FLAGS.setting),		
		('num_models={}', FLAGS.num_models),
		('aug={}', 		FLAGS.augmentation),
		('inf={}', 		FLAGS.inf), 
	]	

elif FLAGS.setting == 'mine': 
	ARGS = dict()
	if FLAGS.dataset == 'cifar10':
		ARGS['eps'] = 0.031
		ARGS['eta'] = 0.007
		ARGS['def_iter'] = 10
		ARGS['attack_method'] = 'MadryEtAl'

	elif FLAGS.dataset == 'cifar100': 
		ARGS['eps'] = 0.01
		ARGS['eta'] = 0.001 
		ARGS['def_iter'] = 10
		ARGS['attack_method'] = 'MadryEtAl'

	else: 
		raise ValueError

	setup = [
		('ds={}',		FLAGS.dataset), 
		('defense={}',	FLAGS.defense), 
		('model={}',	FLAGS.model), 
		('bs={}', 		FLAGS.batch_size),
		('eps={}', 		ARGS['eps']),
		('eta={}', 		ARGS['eta']),
		('def_iter={}', ARGS['def_iter']),
		('num_models={}', FLAGS.num_models),
		('aug={}', 		FLAGS.augmentation),
		('inf={}', 		FLAGS.inf), 
	]	
else: 
	raise ValueError 





model_name = '_'.join([t.format(v) for (t, v) in setup])
print("model name: {}".format(model_name))


# Prepare model model saving directory.
save_dir = os.path.join('checkpoints', model_name)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, 'model.{epoch:03d}.h5')
