from __future__ import print_function
import keras
from keras.layers import Input
from keras.models import Model, load_model

import tensorflow as tf
import cleverhans.attacks as attacks
from cleverhans.utils_tf import model_eval
import matplotlib.pyplot as plt 

from mysetting import *
from model import get_model
from utils import *
from keras_wraper_ensemble import KerasModelWrapper

from lib_attack import gen_adv 
from utils_model import lr_schedule, get_final_checkpoint, get_statistic, get_model_eval
from datasets import Dataset
from lib_method import get_loss
from utils_cm import split_dict, writelog, mkdir_p
from utils_im import plot_historgram, plot_images, plot_prediction
#-----------------------
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

np.random.seed(202005)
tf.set_random_seed(202005)
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
                num_models=1,  # CHANGE HERE, use original label --> model ensemble
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


# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
y = tf.placeholder(tf.float32, shape=(None, num_classes)) # CHANGE HERE, use original label --> model ensemble


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

    model_output = keras.layers.concatenate(model_out)
    model = Model(inputs=model_input, outputs=model_output)
    model_ensemble_output = keras.layers.Average()(model_out)
    model_ensemble = Model(inputs=model_input, outputs=model_ensemble_output)
    wrap_ensemble = KerasModelWrapper(model_ensemble, num_class=num_classes)
    del model_dic, model_out

else: 
    assert(FLAGS.defense=='adv')
    model_input = Input(shape=input_shape)
    model_output = get_model(inputs=model_input, model=FLAGS.model, dataset=FLAGS.dataset)
    model = Model(inputs=model_input, outputs=model_output)
    model_ensemble = model
    wrap_ensemble = KerasModelWrapper(model_ensemble, num_class=num_classes)   




#Load model
model_ensemble.load_weights(filepath)


# Get clean accuracy 
batch_size = 100
eval_par = {'batch_size': batch_size}
preds_nor = model_ensemble(x)
# acc_nor = model_eval(sess, x, y, preds_nor, data.x_test, data.y_test, args=eval_par)
acc_nor, nor_preds = get_model_eval(sess, x, y, preds_nor, data.x_test, data.y_test, batch_size)
print("acc_nor: ", acc_nor)

# Attack list to attack 
attack_list = [
                'MadryEtAl', 
                # 'BasicIterativeMethod', 
                # 'MomentumIterativeMethod', 
                # 'CarliniWagnerL2',
                # 'ElasticNetMethod', 
                'SPSA',
                ] 

nb_samples = 10000

if FLAGS.setting == 'adp': 
    if FLAGS.dataset == 'cifar10':
        eps = 0.031
        eps_iter_range = [0.007]
        nb_iter_range = [250]
        jsma_theta = [0.1, 0.1]
        jsma_gamma = [0.05, 0.1]

elif FLAGS.setting == 'mine': 
    if FLAGS.dataset == 'cifar10':
        eps = 0.031
        eps_iter_range = [0.007]
        nb_iter_range = [250]
        jsma_theta = [0.1, 0.1]
        jsma_gamma = [0.05, 0.1]    


for attack_type in attack_list: 
    if attack_type == 'MadryEtAl': 
        att = attacks.MadryEtAl(wrap_ensemble)
        attack_params = {
            'eps': eps,
            'eps_iter': eps_iter_range,
            'nb_iter': nb_iter_range,
            'ord': np.inf,
            'clip_min': clip_min,
            'clip_max': clip_max,              
        }

    elif attack_type == 'BasicIterativeMethod': 
        att = attacks.BasicIterativeMethod(wrap_ensemble)
        attack_params = {
            'eps': eps,
            'eps_iter': eps_iter_range,
            'nb_iter': nb_iter_range,
            'ord': np.inf,
            'clip_min': clip_min,
            'clip_max': clip_max,              
        }

    elif attack_type == 'MomentumIterativeMethod': 
        att = attacks.MomentumIterativeMethod(wrap_ensemble)
        attack_params = {
            'eps': eps,
            'eps_iter': eps_iter_range,
            'nb_iter': nb_iter_range,
            'ord': np.inf,
            'clip_min': clip_min,
            'clip_max': clip_max,              
        }

    elif attack_type == 'CarliniWagnerL2': 
        att = attacks.CarliniWagnerL2(wrap_ensemble, sess=sess)
        nb_samples = 1000
        attack_params = {
            'batch_size': 100,
            'confidence': 0.1,
            'learning_rate': 0.01,
            'binary_search_steps': 1,
            'max_iterations': 1000,
            'initial_const': 0.1, # c={0.001, 0.01, 0.1}
            'clip_min': clip_min,
            'clip_max': clip_max
        }
    elif attack_type == 'DeepFool': 
        att = attacks.DeepFool(wrap_ensemble, sess=sess)
        attack_params = {
                'overshoot':0.2, 
                'max_iter':100, 
                'nb_candidate':10,
            }   

    elif attack_type == 'ElasticNetMethod': 
        att = attacks.ElasticNetMethod(wrap_ensemble, sess=sess)
        nb_samples = 1000
        attack_params = {
            'batch_size': 100,
            'confidence': 0.1,
            'learning_rate': 0.01,
            'binary_search_steps': 1,
            'max_iterations': 1000,
            'initial_const': 1.0,
            'beta': 1e-2,
            'fista': True,
            'decision_rule': 'EN',
            'clip_min': clip_min,
            'clip_max': clip_max
        }

    elif attack_type == 'SaliencyMapMethod': 
        att = attacks.SaliencyMapMethod(wrap_ensemble, sess=sess)
        attack_params = {
            'batch_size': 100,
            'theta': jsma_theta,
            'gamma': jsma_gamma,
            'clip_min': clip_min,
            'clip_max': clip_max
        }

    elif attack_type == 'SPSA': 
        att = attacks.SPSA(wrap_ensemble, sess=sess)
        nb_samples = 1000
        batch_size = 1
        eval_par = {'batch_size': batch_size}
        attack_params = {
            'eps': eps,
            'clip_min': clip_min,
            'clip_max': clip_max,
            'nb_iter': 50,       
            'y': y,          
        }       

    for att_params in split_dict(attack_params):
        writelog('--------------------', logfile)
        for k in att_params.keys():
            writelog('{}:{}'.format(k, att_params[k]), logfile)

        adv_x = tf.stop_gradient(att.generate(x, **att_params))
        preds_adv = model_ensemble(adv_x)
        # acc_adv = model_eval(sess, x, y, preds_adv, data.x_test[:nb_samples], data.y_test[:nb_samples], args=eval_par)
        acc_adv, adv_preds = get_model_eval(sess, x, y, preds_adv, data.x_test[:nb_samples], data.y_test[:nb_samples], batch_size)

        writelog('attack {}, acc_nor: {:.4f}, acc_adv: {:.4f}, nb_samples: {}'.format(attack_type, acc_nor, acc_adv, nb_samples), logfile)
        # writelog('--------------------', logfile)

        h_nat, m_nat, s_nat = get_statistic(nor_preds)
        h_adv, m_adv, s_adv = get_statistic(adv_preds)

        writelog('Statistic of the prediction of normal: {:.2f}+/-{:.2f}'.format(m_nat, s_nat), logfile)
        writelog('Statistic of the prediction of adv: {:.2f}+/-{:.2f}'.format(m_adv, s_adv), logfile)

        writelog('--------------------', logfile)

        plot_historgram(logdir + '/hist_compare={}.png'.format(attack_type), 
                        h_nat, h_adv, 'Prediction\'s Entropy, attacker={}'.format(attack_type))

        _, eval_x_adv = get_model_eval(sess, x, y, adv_x, data.x_test[:10], data.y_test[:10], 1)
        plot_images(logdir + '/adv_examples_attack={}.png'.format(attack_type), 
                    x=data.x_test[:10]+data.x_train_mean, 
                    y=np.argmax(data.y_test[:10], axis=-1), 
                    x_adv=eval_x_adv+data.x_train_mean, 
                    y_adv=np.argmax(adv_preds[:10], axis=-1))

        for i in range(10):
            plot_prediction(figname=logdir+'/images/predictions_attack={}_idx={}.png'.format(attack_type, i), 
                        image=data.x_test[i]+data.x_train_mean, 
                        adv=eval_x_adv[i]+data.x_train_mean, 
                        pred_nor=nor_preds[i], 
                        pred_adv=adv_preds[i])



