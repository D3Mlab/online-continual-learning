# Online Continual Learning
Official repository of 
* [Online Continual Learning in Image Classification: An Empirical Survey](https://arxiv.org/pdf/2101.10423.pdf) (Under review)
* [Online Class-Incremental Continual Learning with Adversarial Shapley Value](https://arxiv.org/abs/2009.00093) (AAAI 2021)


## Requirements
![](https://img.shields.io/badge/python-3.7-green.svg)

![](https://img.shields.io/badge/torch-1.5.1-blue.svg)
![](https://img.shields.io/badge/torchvision-0.6.1-blue.svg)
![](https://img.shields.io/badge/PyYAML-5.3.1-blue.svg)
![](https://img.shields.io/badge/scikit--learn-0.23.0-blue.svg)
----
Create a virtual enviroment
```sh
virtualenv online-cl
```
Activating a virtual environment
```sh
source online-cl/bin/activate
```
Installing packages
```sh
pip install -r requirements.txt
```

## Datasets 

###Online Class Incremental
- Split CIFAR10
- Split CIFAR100
- CORe50-NC
- Split Mini-ImageNet

###Online Domain Incremental
- NonStationary-MiniImageNet (Noise, Occlusion, Blur)
- CORe50-NI
  
###Data preparation
- CIFAR10 & CIFAR100 will be downloaded during the first run
- CORE50 download: `source fetch_data_setup.sh`
- Mini-ImageNet: Download from https://www.kaggle.com/whitemoon/miniimagenet/download , and place it in datasets/mini_imagenet/
- NonStationary-MiniImageNet will be generated on the fly


## Algorithms 

* ASER: Adversarial Shapley Value Experience Replay(**AAAI, 2021**) [[Paper]](https://arxiv.org/abs/2009.00093)
* EWC++: Efficient and online version of Elastic Weight Consolidation(EWC) (**ECCV, 2018**) [[Paper]](http://arxiv-export-lb.library.cornell.edu/abs/1801.10112)
* iCaRL: Incremental Classifier and Representation Learning (**CVPR, 2017**) [[Paper]](https://arxiv.org/abs/1611.07725)
* LwF: Learning without forgetting (**ECCV, 2016**) [[Paper]](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_37)
* AGEM: Averaged Gradient Episodic Memory (**ICLR, 2019**) [[Paper]](https://openreview.net/forum?id=Hkf2_sC5FX)
* ER: Experience Replay (**ICML Workshop, 2019**) [[Paper]](https://arxiv.org/abs/1902.10486)
* MIR: Maximally Interfered Retrieval (**NeurIPS, 2019**) [[Paper]](https://proceedings.neurips.cc/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html)
* GSS: Gradient-Based Sample Selection (**NeurIPS, 2019**) [[Paper]](https://arxiv.org/pdf/1903.08671.pdf)
* GDumb: Greedy Sampler and Dumb Learner (**ECCV, 2020**) [[Paper]](https://www.robots.ox.ac.uk/~tvg/publications/2020/gdumb.pdf)
* CN-DPM: Continual Neural Dirichlet Process Mixture (**ICLR, 2020**) [[Paper]](https://openreview.net/forum?id=SJxSOJStPr)

## Tricks
- Label trick, [Paper](https://arxiv.org/pdf/1803.10123.pdf)
- Cross entropy with knowledge distillation, [Paper](https://arxiv.org/abs/1807.09536)
- Multiple iterations, [Paper](https://proceedings.neurips.cc/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html)
- Nearest Class Mean classifier, [Paper](https://arxiv.org/abs/2004.00440)
- Separated Softmax, [Paper](https://arxiv.org/abs/2003.13947)
- Review Trick, [Paper](https://arxiv.org/abs/2007.05683)

## Run commands
### Sample commands to run algorithms
```shell
#ER
python general_main.py --data cifar100 --cl_type nc --agent ER --retrieve random --update random --mem_size 5000

#MIR
python general_main.py --data cifar100 --cl_type nc --agent ER --retrieve MIR --update random --mem_size 5000

#GSS
python general_main.py --data cifar100 --cl_type nc --agent ER --retrieve GSS --update random --eps_mem_batch 10 --gss_mem_strength 20 --mem_size 5000

#LwF
python general_main.py --data cifar100 --cl_type nc --agent LWF 

#iCaRL
python general_main.py --data cifar100 --cl_type nc --agent ICARL --retrieve random --update random --mem_size 5000

#EWC++
python general_main.py --data cifar100 --cl_type nc --agent EWC --fisher_update_after 50 --alpha 0.9 --lambda_ 100

#GDumb
python general_main.py --data cifar100 --cl_type nc --agent GDUMB --mem_size 1000 --mem_epoch 30 --minlr 0.0005 --clip 10

#AGEM
python general_main.py --data cifar100 --cl_type nc --agent AGEM --retrieve random --update random --mem_size 5000

#CN-DPM
python general_main.py --data cifar100 --cl_type nc --agent CNDPM --stm_capacity 1000 --classifier_chill 0.01 --log_alpha -300

#ASER
python general_main.py --data cifar100 --cl_type nc --agent ER --update ASER --retrieve ASER --mem_size 5000 --aser_type asvm --n_smp_cls 1.5 --k 3 
```

### Sample commands to run tricks for memory-based methods
```shell
python general_main.py --review_trick True --data cifar100 --cl_type nc --agent ER --retrieve MIR --update random --mem_size 5000 
```


## Repo Structure & Description
    ├──agents                       #Files for different algorithms
        ├──base.py                      #Abstract class for algorithms
        ├──agem.py                      #File for A-GEM
        ├──cndpm.py                     #File for CN-DPM
        ├──ewc_pp.py                    #File for EWC++
        ├──exp_replay.py                #File for ER, MIR and GSS
        ├──gdumb.py                     #File for GDumb
        ├──iCaRL.py                     #File for iCaRL
        ├──lwf.py                       #File for LwF

    ├──continuum                    #Files for create the data stream objects
        ├──dataset_scripts              #Files for processing each specific dataset
            ├──dataset_base.py              #Abstract class for dataset
            ├──cifar10.py                   #File for CIFAR10
            ├──cifar100,py                  #File for CIFAR100
            ├──core50.py                    #File for CORe50
            ├──mini_imagenet.py             #File for Mini_ImageNet
            ├──openloris.py                 #File for OpenLORIS
        ├──continuum.py             
        ├──data_utils.py
        ├──non_stationary.py
    
    ├──models                       #Files for backbone models
        ├──ndpm                         #Files for models of CN-DPM 
            ├──...
        ├──pretrained.py                #Files for pre-trained models
        ├──resnet.py                    #Files for ResNet

    ├──utils                        # Files for utilities
        ├──buffer                       #Files related to buffer
            ├──aser_retrieve.py
            ├──aser_update.py
            ├──aser_utils.py
            ├──buffer.py
            ├──buffer_utils.py
            ├──gss_greedy_update.py
            ├──mir_retrieve.py
            ├──random_retrieve.py
            ├──reservoir_update.py
        ├──global_vars.py               #Global variables for CN-DPM
        ├──io.py                        #Code related to load and store csv or yarml
        ├──kd_manager.py                #File for knowledge distillation
        ├──name_match.py                # 
        ├──setup_elements.py
        ├──utils.py

    ├──
        ├──
        ├──
        ├──




    ├──
    ├──
    ├──
    ├──
    ├──
    ├──
    ├──
    ├──










        

## Citation 
If you use this paper/code in your research, please consider citing us:

**Online Continual Learning in Image Classification: An Empirical Survey**

Under review, preprint on arXiv [here](https://arxiv.org/pdf/2101.10423.pdf).
```
@article{mai2021online,
  title={Online Continual Learning in Image Classification: An Empirical Survey},
  author={Mai, Zheda and Li, Ruiwen and Jeong, Jihwan and Quispe, David and Kim, Hyunwoo and Sanner, Scott},
  journal={arXiv preprint arXiv:2101.10423},
  year={2021}
}
```

**Online Class-Incremental Continual Learning with Adversarial Shapley Value**

[Accepted at AAAI2021](https://arxiv.org/abs/2009.00093)
```
@article{shim2020online,
  title={Online Class-Incremental Continual Learning with Adversarial Shapley Value},
  author={Shim, Dongsub and Mai, Zheda and Jeong, Jihwan and Sanner, Scott and Kim, Hyunwoo and Jang, Jongseong},
  journal={arXiv preprint arXiv:2009.00093},
  year={2020}
}
```