from __future__ import print_function
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model

import tensorflow as tf
import matplotlib.pyplot as plt 

from mysetting import *
from model import get_model

from utils_model import get_final_checkpoint, get_statistic
from datasets import Dataset
from utils_cm import writelog, mkdir_p
from utils_im import plot_historgram, plot_images, plot_prediction

import torch 

sys.path.append('/home/ethan/anhbui_phd/auto-attack/')
sys.path.append('/home/ethan/anhbui_phd/auto-attack/autoattack/')
import utils_tf2
from autoattack import AutoAttack

#-----------------------
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

    model_output = keras.layers.concatenate(model_out)
    model = Model(inputs=model_input, outputs=model_output)
    model_ensemble = keras.layers.Average()(model_out)
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


# Get clean accuracy 
eps = 0.031
bs = 10

writelog('--------------------', logfile)
writelog('attack_type:{}'.format("auto-attack"), logfile)
writelog('version:{}'.format('standard'), logfile)
writelog('eps:{}'.format(eps), logfile)

""" Init attacker 
    Robustness Evaluation using Auto attack
    Ref: https://github.com/fra31/auto-attack
"""
nb_samples = 10
x_test = data.x_test[:nb_samples]
y_test = data.y_test[:nb_samples]



model_adapted = utils_tf2.ModelAdapter(model_ensemble)

adversary = AutoAttack(model_adapted, norm='Linf', eps=eps, version='standard', is_tf_model=True, log_path=logfile)

# Convert to required format: NHWC --> NCHW
_x = torch.from_numpy(np.moveaxis(x_test,3,1)) 
_y = torch.from_numpy(np.argmax(y_test, axis=-1))
_x.float().to("cuda")
_y.to("cuda")

x_adv = adversary.run_standard_evaluation(_x, _y, bs=bs)
x_adv = x_adv.cpu().numpy()

# Convert back to NHWC 
x_adv = np.moveaxis(x_adv, 1, 3)


pred_adv = model_ensemble.predict(x=x_adv)
pred_nat = model_ensemble.predict(x=x_test)
acc_nat_test = np.mean(np.argmax(pred_nat, axis=-1)==np.argmax(y_test, axis=-1))
acc_adv_test = np.mean(np.argmax(pred_adv, axis=-1)==np.argmax(y_test, axis=-1))

h_nat, m_nat, s_nat = get_statistic(pred_nat)
h_adv, m_adv, s_adv = get_statistic(pred_adv)

writelog('attack {}, acc_nat_test: {:.4f}, acc_adv_test: {:.4f}, nb_samples: {}'.format(
    'auto-attack', acc_nat_test, acc_adv_test, nb_samples), logfile)

writelog('Statistic of the prediction of normal: {:.2f}+/-{:.2f}'.format(m_nat, s_nat), logfile)
writelog('Statistic of the prediction of adv: {:.2f}+/-{:.2f}'.format(m_adv, s_adv), logfile)

writelog('--------------------', logfile)

plot_historgram(logdir + '/hist_compare={}.png'.format('auto-attack'), 
                    h_nat, h_adv, 'Prediction\'s Entropy, attacker={}'.format('auto-attack'))

plot_images(logdir + '/adv_examples_attack={}.png'.format('auto-attack'), 
            x=x_test[:10]+data.x_train_mean, 
            y=np.argmax(y_test[:10], axis=-1), 
            x_adv=x_adv[:10]+data.x_train_mean, 
            y_adv=np.argmax(pred_adv[:10], axis=-1))

for i in range(10):
    plot_prediction(figname=logdir+'/images/predictions_attack={}_idx={}.png'.format('auto-attack', i), 
                image=x_test[i]+data.x_train_mean, 
                adv=x_adv[i]+data.x_train_mean, 
                pred_nor=pred_nat[i], 
                pred_adv=pred_adv[i])
