import subprocess 
import numpy as np 
import os 
import sys

os.chdir('./')
ST = 'python '

stand = dict()
conf = dict()

stand = dict()
stand['dataset'] = 'cifar10' 
stand['batch_size'] = 64
stand['defense'] = 'none'
stand['model'] = 'resnet20'
stand['num_epochs'] = 180
stand['num_models'] = 2
stand['augmentation'] = True
stand['setting'] = 'adp'


# conf['none'] = stand.copy()
# conf['none']['defense'] = 'none'

# conf['none3'] = stand.copy()
# conf['none3']['defense'] = 'none'
# conf['none3']['num_models'] = 3

# conf['adv'] = stand.copy()
# conf['adv']['defense'] = 'adv'
# conf['adv']['num_models'] = 1

# conf['adv_en'] = stand.copy()
# conf['adv_en']['defense'] = 'adv_en'

# conf['adv_en3'] = stand.copy()
# conf['adv_en3']['defense'] = 'adv_en'
# conf['adv_en3']['num_models'] = 3

# conf['adp'] = stand.copy()
# conf['adp']['defense'] = 'adp'

# conf['adp3'] = stand.copy()
# conf['adp3']['defense'] = 'adp'
# conf['adp3']['num_models'] = 3

# conf['adp3d'] = stand.copy()
# conf['adp3d']['defense'] = 'adp'
# conf['adp3d']['num_models'] = 3
# conf['adp3d']['inf'] = 'downloaded'

# conf['pie0'] = stand.copy()
# conf['pie0']['defense'] = 'pie'
# conf['pie0']['wdm'] = 0
# conf['pie0']['inf'] = 'wdm=0.0'

# conf['pie30'] = stand.copy()
# conf['pie30']['defense'] = 'pie'
# conf['pie30']['wdm'] = 0
# conf['pie30']['num_models'] = 3
# conf['pie30']['inf'] = 'wdm=0.0'

# conf['pie1'] = stand.copy()
# conf['pie1']['defense'] = 'pie'
# conf['pie1']['wdm'] = 1
# conf['pie1']['inf'] = 'wdm=1.0'

conf['pie5'] = stand.copy()
conf['pie5']['defense'] = 'pie'
conf['pie5']['wce'] = 0
conf['pie5']['wdm'] = 5
conf['pie5']['inf'] = 'wce=0.0_wdm=5.0'

skip = ['_', '_', '_', '_']

for k in list(conf.keys()):
	if k in skip: 
		continue

	# chST = 'main_train.py '
	chST = 'main_eval_mul_attacks.py '
	# chST = 'main_eval_bbattack.py '
	# chST = 'main_eval_auto_attack.py '

	exp = conf[k]
	sub = ' '.join(['--{}={}'.format(t, exp[t]) for t in exp.keys()])
	print('***', sub)
	subprocess.call([ST + chST + sub], shell=True)

	# # chST = 'main_train.py '
	# # chST = 'main_eval_mul_attacks.py '
	# # chST = 'main_eval_bbattack.py '
	# # chST = 'main_eval_auto_attack.py '

	# exp = conf[k]
	# sub = ' '.join(['--{}={}'.format(t, exp[t]) for t in exp.keys()])
	# print('***', sub)
	# subprocess.call([ST + chST + sub], shell=True)

	# # chST = 'main_train.py '
	# # chST = 'main_eval_mul_attacks.py '
	# # chST = 'main_eval_bbattack.py '
	# # chST = 'main_eval_auto_attack.py '

	# exp = conf[k]
	# sub = ' '.join(['--{}={}'.format(t, exp[t]) for t in exp.keys()])
	# print('***', sub)
	# subprocess.call([ST + chST + sub], shell=True)