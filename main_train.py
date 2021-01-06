from __future__ import print_function
import keras
from keras.layers import Input
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from keras.callbacks import CSVLogger


import tensorflow as tf
import numpy as np
import os
import sys
import time

from mysetting import *
from model import get_model
from utils import *
from keras_wraper_ensemble import KerasModelWrapper

from lib_attack import gen_adv 
from utils_model import lr_schedule
from utils_cm import backup, writelog
from datasets import Dataset
from lib_method import get_loss

#-----------------------
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

np.random.seed(202005)
tf.set_random_seed(202005)
#-----------------------

# Backup code to local 
logdir = os.path.join('log', model_name)
codedir = os.path.join(logdir, 'codes')
if not os.path.isdir(logdir):
    os.makedirs(logdir)
    os.makedirs(codedir)

backup('./', codedir)
csv_logger = CSVLogger(logdir+'/log.csv', append=True, separator=';')
logfile = logdir + '/logtime.txt'

# Training parameters
epochs = FLAGS.num_epochs

""" Get data 
    Args: 
        dataset: mnist, cifar10, cifar100 
        num_models: number of models, to duplicate label 
        subtract_pixel_mean: flag to subtract with mean of x train 
        clip_min: 
        clip_max 
    Returns: 
        data: class 
        data.x_train, data.y_train, data.x_test, data.y_test 
        data.datagen: generator 
"""
data = Dataset(ds=FLAGS.dataset, 
                num_models=FLAGS.num_models, 
                subtract_pixel_mean=True, 
                clip_min=0.0, 
                clip_max=1.0)

# retrieve infor 
clip_min = data.clip_min
clip_max = data.clip_max
x_train_mean = data.x_train_mean 
input_shape = data.x_shape

print('subtract_pixel_mean, mean={}, clip_min={}, clip_max={}'.format(
    x_train_mean, clip_min, clip_max))


""" Build an ensemble model 
    Args: 
        input_shape: 
        num_models: 
        [Not use] model_type: resnet20, cnn, nonwide (future), wideresnet (future) 
    Returns: 
        model: [pred_model_1, pred_model_2, ...]
        model_ensemble: pred_model_en # where pred_model_en = average([pred_model_1, pred_model_2, ...])
        model_dic: keras model dic {'0': model_0, '1': model_1, ...} 
"""

if FLAGS.num_models > 1:
    model_input = Input(shape=input_shape)
    model_dic = {}
    model_out = []
    for i in range(FLAGS.num_models):
        # resnet_v1 return: model, inputs, outputs, logits, final_features
        model_i_output = get_model(inputs=model_input, model=FLAGS.model, dataset=FLAGS.dataset)
        model_out.append(model_i_output)
        model_dic[i] = Model(inputs=model_input, outputs=model_i_output)

    model_output = keras.layers.concatenate(model_out)
    model = Model(inputs=model_input, outputs=model_output)
    model_ensemble = keras.layers.Average()(model_out)
    model_ensemble = Model(inputs=model_input, outputs=model_ensemble)
    wrap_ensemble = KerasModelWrapper(model_ensemble, num_class=num_classes)
else: 
    assert(FLAGS.defense=='adv')
    model_input = Input(shape=input_shape)
    model_output = get_model(inputs=model_input, model=FLAGS.model, dataset=FLAGS.dataset)
    model = Model(inputs=model_input, outputs=model_output)
    wrap_ensemble = KerasModelWrapper(model, num_class=num_classes)    

""" Generate adversarial examples
    Args: 
        eps 

    Return: 
        adv_x: adv of model_ensemble 
        adv_x1: adv of model_1 
        adv_x2: adv of model_2
    
    Note: We use two setting here:
    # ADP Adversarial training setting is fixed as ADP paper
    # eps = tf.random_uniform((), 0.01, 0.05)
    # eta = eps/10.
    # my setting is flexible based on FLAGS 
"""

adv_x = gen_adv(wrap_model=wrap_ensemble,
                model_input=model_input, 
                attack_method=ARGS['attack_method'], 
                eps=ARGS['eps'], 
                eta=ARGS['eta'], 
                def_iter=ARGS['def_iter'],
                clip_min=clip_min, 
                clip_max=clip_max)
print("Finish generate adv of ensemble")


adv_pred_concat = model(adv_x)
nor_pred_concat = model(model_input)

""" Get crossing prediction output f_j(adv_xi)
    Args: 
        
    Return: 
        adv_pred[j][i] = model_dic[j](adv_xes[i])
        nor_pred[j] = model_dic[j](model_input)

"""
if FLAGS.defense in ['cce']:
    # Generate adversarial example of individual network 
    adv_xes = {}
    for i in range(FLAGS.num_models): 
        wrap_i = KerasModelWrapper(model_dic[i], num_class=num_classes)
        adv_xes[i] = gen_adv(wrap_model=wrap_i,
                                model_input=model_input, 
                                attack_method=ARGS['attack_method'], 
                                eps=ARGS['eps'], 
                                eta=ARGS['eta'], 
                                def_iter=ARGS['def_iter'],
                                clip_min=clip_min, 
                                clip_max=clip_max)
        print("Finish generate adv of model {}".format(i))

    # Generate cross prediction 
    adv_pred = dict()
    nor_pred = dict()
    for i in range(FLAGS.num_models): 
        nor_pred[i] = model_dic[i](model_input)
        adv_pred['m'+str(i)] = dict()
        for j in range(FLAGS.num_models): 
            adv_pred['m'+str(i)]['x'+str(j)] = model_dic[i](adv_xes[j])
            print("Finish generate prediction model {} with adv {}".format(i, j))
else: 
    adv_pred = None 
    nor_pred = None

""" Get objective function 
    Args: 
        method 
    Return: 
        adv_loss if method == 'adv_en'
        adv_EEDPP if method == 'adp'
        adv_CCE if method == 'cce'

"""
def adv_acc_metric(y_true, y_pred, num_models=FLAGS.num_models):
    y_p = tf.split(adv_pred_concat, num_models, axis=-1)
    y_t = tf.split(y_true, num_models, axis=-1)
    acc = 0
    for i in range(num_models):
        acc += tf.keras.metrics.categorical_accuracy(y_t[i], y_p[i])
    return acc / num_models

model.compile(
    loss=get_loss(FLAGS.defense, adv_pred_concat=adv_pred_concat, nor_pred_concat=nor_pred_concat, 
                adv_pred=adv_pred, nor_pred=nor_pred),
    optimizer=Adam(lr=lr_schedule(0)),
    metrics=[acc_metric, adv_acc_metric])
model.summary()
print("*** Finish compling ***")
print("model metrics_names: ", model.metrics_names)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(
    filepath=filepath,
    monitor='val_adv_acc_metric',
    mode='max',
    verbose=2,
    save_best_only=False) # CHANGE HERE to keep the model with the latest model

lr_scheduler = LearningRateScheduler(lr_schedule)

callbacks = [checkpoint, lr_scheduler, csv_logger]


# Run training, with or without data augmentation.
if not FLAGS.augmentation:
    print('Not using data augmentation.')
    model.fit(
        data.x_train,
        data.y_train,
        batch_size=FLAGS.batch_size,
        epochs=epochs,
        validation_data=(data.x_test, data.y_test),
        shuffle=True,
        verbose=1,
        callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(
        data.datagen.flow(data.x_train, data.y_train, batch_size=FLAGS.batch_size),
        validation_data=(data.x_test, data.y_test),
        steps_per_epoch=np.shape(data.x_train)[0]//FLAGS.batch_size,
        epochs=epochs,
        verbose=1,
        workers=4,
        callbacks=callbacks)

model.save(save_dir+'/model-final.h5')
