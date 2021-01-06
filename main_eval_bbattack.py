import os
os.environ["GIT_PYTHON_REFRESH"]="quiet" 
import numpy as np

import sys
sys.path.append("/home/ethan/anhbui_phd/foolbox/")
import foolbox as fb

from mysetting import *
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model

print('tf version: ', tf.__version__)

from model import get_model

from utils_model import get_final_checkpoint
from datasets import Dataset
from utils_cm import writelog, mkdir_p
from utils_im import plot_images, plot_prediction
#-----------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

np.random.seed(202005)
tf.random.set_seed(202005)
#-----------------------

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
                num_models=1, # CHANGE HERE, use original label --> model ensemble
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

""" Load pretrained model
    Args: 
        save_dir: global variable in mysetting 

    Returns: 
        final_epoch = get_final_epoch(save_dir, selected_epoch=None)
        filepath = os.path.join(save_dir, 'model.{epoch:03d}.h5'.format(final_epoch))
"""
filepath = get_final_checkpoint(save_dir, selected_epoch=FLAGS.epoch)
print('Restore model checkpoints from %s'% filepath)

logdir = os.path.join('log', model_name)
mkdir_p(logdir)
mkdir_p(logdir+'/images/')
logfile = logdir + '/log_eval_attacks.txt'
writelog("model name: {}".format(model_name), logfile)

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

    model_output = tf.keras.layers.concatenate(model_out)
    model = Model(inputs=model_input, outputs=model_output)
    model_ensemble = tf.keras.layers.Average()(model_out)
    model_ensemble = Model(inputs=model_input, outputs=model_ensemble)
    del model_dic, model_out

else: 
    assert(FLAGS.defense=='adv')
    model_input = Input(shape=input_shape)
    model_output = get_model(inputs=model_input, model=FLAGS.model, dataset=FLAGS.dataset)
    model = Model(inputs=model_input, outputs=model_output)
    model_ensemble = model


#Load model
model_ensemble.load_weights(filepath)

nb_samples = 20

x_test = data.x_test[:nb_samples]
y_test = data.y_test[:nb_samples]
y_test = np.argmax(y_test, axis=1).flatten()

pred = model_ensemble.predict(x_test).argmax(1)
acc_nor = np.mean(pred == y_test)


# convert to Foolbox model
fmodel = fb.models.TensorFlowModel(model_ensemble, bounds=(-2, 2))


images = tf.convert_to_tensor(x_test, dtype=tf.float32)
labels = tf.convert_to_tensor(y_test)

# a simple wrapper for the init attack in BB
class init_attack(object):
    
    def __init__(self, attack):
        self.attack = attack
        
    def run(self, model, originals, criterion_):
        return self.attack(model, images, criterion=criterion_, epsilons=0.3)[1]




writelog("------- Evaluation using foolbox -------", logfile)
attack_list = [
                # 'LinfPGD20',
                # 'LinfPGD50',
                # 'LinfPGD250', 
                'LinfinityBrendelBethgeAttack', 
                ] 

if FLAGS.setting == 'adp': 
    eps = 0.031
    eta = 0.007 
elif FLAGS.setting == 'mine': 
    eps = 0.031
    eta = 0.007 

repetitions = 3

batch_size = 20 
nb_batch = np.shape(x_test)[0] // batch_size

for attack_type in attack_list: 
    acc_adv = 0
    total_images = 0   

    if attack_type == 'LinfPGD20': 
        steps = 20
        atk = fb.attacks.LinfPGD(steps=steps, abs_stepsize=eta, random_start=True)    
    elif attack_type == 'LinfPGD50': 
        steps = 50
        atk = fb.attacks.LinfPGD(steps=steps, abs_stepsize=eta, random_start=True)
    elif attack_type == 'LinfPGD250': 
        steps = 250 
        atk = fb.attacks.LinfPGD(steps=steps, abs_stepsize=eta, random_start=True)   
    elif attack_type == 'LinfinityBrendelBethgeAttack': 
        steps = 100
        pdg_init_attack = fb.attacks.LinfPGD(steps=20, abs_stepsize=eta, random_start=True)
        atk = fb.attacks.LinfinityBrendelBethgeAttack(init_attack(pdg_init_attack), steps=steps)          

    for _images, _labels in zip(np.split(x_test, nb_batch), np.split(y_test, nb_batch)):
        mask = np.array([True] * batch_size)
        images = tf.convert_to_tensor(_images[mask], dtype=tf.float32)
        labels = tf.convert_to_tensor(_labels[mask])

        
        for r in range(repetitions):
            if mask.sum() > 0:
                adv, adv_clipped, adv_mask = atk(fmodel, images, 
                	criterion=fb.criteria.Misclassification(labels), epsilons=eps)
            
                mask[mask] = ~adv_mask.numpy()

                images = tf.convert_to_tensor(_images[mask], dtype=tf.float32)
                labels = tf.convert_to_tensor(_labels[mask])
            
            if r == 0: 
                x_adv = adv 
        acc_adv += (1 - adv_mask.numpy().mean()) * len(adv)
        total_images += _images.shape[0]
        
        writelog("total_images={}, acc_nor={}, acc_adv={}".format(total_images, acc_nor, acc_adv / total_images), logfile)
    acc_adv = acc_adv / total_images

    writelog("attack_type={}, repetitions={}, eps={}, eta={}, steps={}, total_images={}".format(attack_type, 
        repetitions, eps, eta, steps, total_images), logfile)
    writelog("acc_nor={}, acc_adv={}".format(acc_nor, acc_adv), logfile)


    xb = x_test[:10]
    x_adv = x_adv[:10]
    yb = y_test[:10]
    print(np.shape(xb))
    print(np.shape(yb))

    pred_nat = fmodel(xb)
    pred_adv = fmodel(x_adv)
    y_adv = np.argmax(pred_adv, axis=-1)
    plot_images(logdir + '/adv_examples_attack={}.png'.format(attack_type), 
                x=xb+data.x_train_mean, 
                y=yb, 
                x_adv=x_adv+data.x_train_mean, 
                y_adv=y_adv)

    for i in range(10):
        plot_prediction(figname=logdir+'/images/predictions_attack={}_idx={}.png'.format(attack_type, i), 
                image=xb[i]+data.x_train_mean, 
                adv=x_adv[i]+data.x_train_mean, 
                pred_nor=pred_nat[i], 
                pred_adv=pred_adv[i])


writelog("------- Finish Evaluation using foolbox -------", logfile)    




