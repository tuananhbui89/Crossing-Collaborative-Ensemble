# Crossing Collaborative Ensemble

The implementation of the Crossing Collaborative Ensemble (CCE) algorithm, which helps to improve the adversarial robustness of an ensemble model. More detail, please refer to our paper:

["Improving Ensemble Robustness by Collaboratively Promoting and Demoting Adversarial Robustness"](https://arxiv.org/abs/2009.09612)</br>
**A. Bui**, T. Le, H. Zhao, P. Montague, O. de Vel, T. Abraham, D. Phung.</br>
Proceedings of the AAAI Conference on Artificial Intelligence 2021.

## Requirements

This project requires two separate environments which are TF1 and TF2 

(A) The TF1 enviroment for training the model and evaluation using cleverhans lib 
- Python: 3.7
- cleverhans: 3.0.1
- Keras: 2.2.4
- tensorflow-gpu: 1.15.0

(B) The TF2 eviroment for evaluation using [foolbox](https://foolbox.readthedocs.io/en/stable/) lib which mainly for Brendel & Bethge Attack
- Python: 3.7git
- foolbox: 3.0.4
- tensorflow-gpu: 2.0.0
- torch: 1.5.0

## Baselines 
We also provide the implementation of the baseline methods in our paper, which are: 
- ADV: PGD Adversarial Training on a single model 
- ADV_EN: PGD Adversarial Training on an ensemble model 
- ADP: Adaptive Diversity Promoting with lamba=2.0 and log_det_lamda=0.5 which is the best setting as reported in their paper 

## Training baselines and CCE (Using TF1)

We provide the default setting for each corresponding dataset (CIFAR10, CIFAR100) which is used in our paper. For training method A on dataset D with an ensemble of N models, just run the following script. 
```shell
python main_train.py --defense=A --num_models=N --dataset=D --model=M
```
where
- defense={'adv', 'adv_en', 'adp', 'cce'}
- dataset={'cifar10', 'cifar100'}
- num_models={1, 2, 3}
- model={'cnn', 'resnet20'}

## Evaluation codes

For most attacks except B&B attack, we will use the TF1 enviroment to run the evaluation. 
```shell
python main_eval_mul_attacks.py --defense=A --num_models=N --dataset=D --model=M
```

For B&B attack, we need to activate TF2 environment and run the following script.
```shell
python main_eval_bbattack.py --defense=A --num_models=N --dataset=D --model=M
```

For Auto attack, we need to activate TF2 environment and run the following script
```shell
python main_eval_auto_attack.py --defense=A --num_models=N --dataset=D --model=M
```

### Note
<!-- - B&B attack: We argue against the parameter setting in the paper ["On Adaptive Attacks to Adversarial Example Defenses"](https://arxiv.org/abs/2002.08347) to evaluate the ADP method. It is because the ADP training method used epsilon=U(0.01,0.05) while B&B attack in this paper used the PGD with epsilon=0.15, k=20 as an initialization attack which is very strong attack strength.    -->
- From my experience, B&B attack usually fail in the initialization attack, to run it smoothly, we modify in the foolbox lib by comment out the line 436 in file brendel_bethge.py 
```shell
    # assert is_adversarial(best_advs).all()
```
## Cite
Please cite our paper if you find this work useful for your research

    @inproceedings{bui2020improving,
        title={Improving Ensemble Robustness by Collaboratively Promoting and Demoting Adversarial Robustness},
        author={Bui, Anh and Le, Trung and Zhao, He and Montague, Paul and deVel, Olivier and Abraham, Tamas and Phung, Dinh},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence 2021},
        year={2020}
    }

## References
- Our implementation is widely adapted from [ADP](https://github.com/P2333/Adaptive-Diversity-Promoting) implementation. 
- The B&B attack is adapted from [the implementation](https://github.com/wielandbrendel/adaptive_attacks_paper/tree/master/07_ensemble_diversity) of the paper ["On Adaptive Attacks to Adversarial Example Defenses"](https://arxiv.org/abs/2002.08347). 
- The Auto-attack is adapted from [the implementation](https://github.com/fra31/auto-attack) of the paper ["Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks", Francesco Croce, Matthias Hein, ICML 2020](https://arxiv.org/abs/2003.01690).