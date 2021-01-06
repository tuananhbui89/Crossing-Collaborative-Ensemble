import numpy as np 
import tensorflow as tf
import keras 
from utils import *  
from mysetting import *
from functools import partial 

def get_loss(method, adv_pred_concat, nor_pred_concat, adv_pred, nor_pred): 
    if method == 'none': 
        return adv_NONE(nor_pred_concat)
    elif method == 'adp': 
        return adv_ADP(adv_pred_concat, nor_pred_concat)
    elif method == 'adv': 
        assert(FLAGS.num_models==1)
        return adv_EN(adv_pred_concat, nor_pred_concat)
    elif method == 'adv_en': 
        return adv_EN(adv_pred_concat, nor_pred_concat)
    elif method == 'cce': 
        return adv_CCE(adv_pred, nor_pred)


def adv_ADP(adv_pred_concat, nor_pred_concat): 
    def adv_EEDPP(_y_true, _y_pred):
        return _Loss_withEE_DPP(_y_true, adv_pred_concat) + _Loss_withEE_DPP(_y_true, nor_pred_concat)

    def _Loss_withEE_DPP(y_true,
                        y_pred,
                        num_models=FLAGS.num_models,
                        label_smooth=FLAGS.label_smooth):

        scale = (1 - label_smooth) / (num_classes * label_smooth - 1)
        y_t_ls = scale * tf.ones_like(y_true) + y_true
        y_t_ls = (num_models * y_t_ls) / tf.reduce_sum(y_t_ls, axis=1, keepdims=True)
        y_p = tf.split(y_pred, num_models, axis=-1)
        y_t = tf.split(y_t_ls, num_models, axis=-1)
        CE_all = 0
        for i in range(num_models):
            CE_all += keras.losses.categorical_crossentropy(y_t[i], y_p[i])
        EE = Ensemble_Entropy(y_true, y_pred, num_models)
        log_dets = log_det(y_true, y_pred, num_models)
        return CE_all - FLAGS.lamda * EE - FLAGS.log_det_lamda * log_dets

    return adv_EEDPP

def adv_EN(adv_pred_concat, nor_pred_concat): 
    def adv_CE(_y_true, _y_pred): 
        return _Loss_CE(_y_true, adv_pred_concat) + _Loss_CE(_y_true, nor_pred_concat)

    def _Loss_CE(y_true, y_pred, num_models=FLAGS.num_models): 
        y_p = tf.split(y_pred, num_models, axis=-1)
        y_t = tf.split(y_true, num_models, axis=-1)
        CE_all = 0 
        for i in range(num_models): 
            CE_all += keras.losses.categorical_crossentropy(y_t[i], y_p[i])

        return CE_all 

    return adv_CE 

def adv_NONE(nor_pred_concat): 
    def adv_CE(_y_true, _y_pred): 
        return _Loss_CE(_y_true, nor_pred_concat)

    def _Loss_CE(y_true, y_pred, num_models=FLAGS.num_models): 
        y_p = tf.split(y_pred, num_models, axis=-1)
        y_t = tf.split(y_true, num_models, axis=-1)
        CE_all = 0 
        for i in range(num_models): 
            CE_all += keras.losses.categorical_crossentropy(y_t[i], y_p[i])

        return CE_all 

    return adv_CE 

def adv_CCE(adv_pred, nor_pred, num_models=FLAGS.num_models, wce=FLAGS.wce, wdm=FLAGS.wdm, wpm=FLAGS.wpm): 
    """
    crossing collaborative ensemble algorithm 
        using predicted label to decide a next action: 
            if argmax(_y_pred) == argmax(_y_true):
                promoting model_i with adversarial example x_j
            else: 
                demoting model_i with adversarial example x_j
    """
    def cceloss(_y_true, _y_pred, return_loss_i=False): 
        true_y = tf.split(_y_true, num_models, axis=-1)
        loss_i = dict()
        for i in range(num_models): 
            CE_nor_i = tf.keras.losses.categorical_crossentropy(true_y[i], nor_pred[i])
            CE_adv_i = tf.keras.losses.categorical_crossentropy(true_y[i], adv_pred['m'+str(i)]['x'+str(i)])
            loss_i[i] = CE_nor_i + wce * CE_adv_i
            for j in range(num_models): 
                if j != i: 
                    w = _fil_pred(true_y, adv_pred['m'+str(i)]['x'+str(j)]) # [batch_size, ]
                    cce = tf.nn.softmax_cross_entropy_with_logits_v2(adv_pred['m'+str(i)]['x'+str(j)], 
                                                                adv_pred['m'+str(i)]['x'+str(j)]) # [batch_size, ]

                    ce = tf.nn.softmax_cross_entropy_with_logits_v2(true_y[i], 
                                                                adv_pred['m'+str(i)]['x'+str(j)]) # [batch_size, ]

                    if wdm != 0: 
                        loss_i[i] -= wdm/(num_models-1)*tf.reduce_mean(tf.multiply(cce, 1.0000001 - w), axis=0)
                    if wpm != 0:
                        loss_i[i] += wpm/(num_models-1)*tf.reduce_mean(tf.multiply(ce, w), axis=0)

        loss_a = 0 
        for i in range(num_models): 
            loss_a += loss_i[i] 
        
        if return_loss_i: 
            return loss_a, loss_i 
        else:
            return loss_a 

    def _fil_pred(y_true, y_pred): 
        # return the probability of the true index 
        t = tf.multiply(y_true, y_pred)
        return tf.reduce_sum(t, axis=-1)

    return cceloss