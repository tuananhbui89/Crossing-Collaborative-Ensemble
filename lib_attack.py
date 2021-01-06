import tensorflow as tf 
import cleverhans.attacks as attacks

def gen_adv(wrap_model, model_input, attack_method, eps, eta, def_iter,
    clip_min=0., clip_max=1.):

    """
    Generate adversarial examples using keras wrapper 
    """
    
    if attack_method == 'MadryEtAl':
        att = attacks.MadryEtAl(wrap_model)
        att_params = {
            'eps': eps,
            'eps_iter': eta,
            'clip_min': clip_min,
            'clip_max': clip_max,
            'nb_iter': def_iter
        }
    elif attack_method == 'MomentumIterativeMethod':
        att = attacks.MomentumIterativeMethod(wrap_model)
        att_params = {
            'eps': eps,
            'eps_iter': eta,
            'clip_min': clip_min,
            'clip_max': clip_max,
            'nb_iter': def_iter
        }
    elif attack_method == 'FastGradientMethod':
        att = attacks.FastGradientMethod(wrap_model)
        att_params = {'eps': eps,
                       'clip_min': clip_min,
                       'clip_max': clip_max}

    print('attack_method: {}'.format(attack_method))     
    for k in att_params.keys():       
    	print('{}:{}'.format(k,att_params[k]))       
    adv_x = tf.stop_gradient(att.generate(model_input, **att_params))

    return adv_x 