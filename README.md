# LSHFM.multiclassification

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/in-defense-of-feature-mimicking-for-knowledge/knowledge-distillation-on-imagenet)](https://paperswithcode.com/sota/knowledge-distillation-on-imagenet?p=in-defense-of-feature-mimicking-for-knowledge)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/in-defense-of-feature-mimicking-for-knowledge/knowledge-distillation-on-coco)](https://paperswithcode.com/sota/knowledge-distillation-on-coco?p=in-defense-of-feature-mimicking-for-knowledge)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/in-defense-of-feature-mimicking-for-knowledge/knowledge-distillation-on-pascal-voc)](https://paperswithcode.com/sota/knowledge-distillation-on-pascal-voc?p=in-defense-of-feature-mimicking-for-knowledge)

This is the PyTorch source code for [Distilling Knowledge by Mimicking Features](https://arxiv.org/abs/2011.01424). We provide all codes for three tasks.

* single-class image classification: [LSHFM.singleclassification](https://github.com/DoctorKey/LSHFM.singleclassification)
* multi-class image classification: [LSHFM.multiclassification](https://github.com/DoctorKey/LSHFM.multiclassification)
* object detection: [LSHFM.detection](https://github.com/DoctorKey/LSHFM.detection)

## Usage

### Install the dependencies

The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands:

```
conda create -n pytorch python=3.6
conda activate pytorch
conda install pytorch torchvision cudatoolkit=10.0 
conda install pycocotools
```

### Prepare the datasets

#### VOC2007

Organize the dataset as below. Then check and edit the `voc2007` function in `src/dataset/datasets.py`. 
```
.
├── VOCdevkit
│   └── VOC2007
├── VOCtest_06-Nov-2007.tar
└── VOCtrainval_06-Nov-2007.tar
```

#### COCO2014

Organize the dataset as below. Then check and edit the `COCO2014` function in `src/dataset/datasets.py`. 
```
├── annotations
│   ├── instances_train2014.json
│   └── instances_val2014.json
├── train2014
└── val2014
```

### Run

Please check scripts in `experiments`. The scripts save the values of hyperparamters we used. You just need to adjust `gpus` according to your system.

#### Baseline

You can run the baseline by the command:
```
python experiments/[data]/[network]/baseline_BCE.py
e.g.
python experiments/voc2007/resnet34/baseline_BCE.py
```

When training, the log and checkpoint will be saved in `results`. After training, you can move the teacher's checkpoint into `pretrained` and renamed it as `resnet34_voc2007_BCE_91.69.ckpt`. This checkpoint is needed to initialize the teacher for feature mimicking.

#### Feature mimicking

As in our paper, feature mimicking consists two stages. The 1st stage aims at aligning two networks' feature spaces.

The command of the 1st stage is
```
python experiments/[data]/[student network]/KD_LSHL2_train_feat_fc.py
e.g.
python experiments/voc2007/resnet18/KD_LSHL2_train_feat_fc.py
```

After training, you can move the saved checkpoint into `pretrained` and renamed it as `r18_feat_fc@r34_voc_lshl2_1.ckpt`. This checkpoint is needed to initialize the student in the 2nd stage.

The command of the 2nd stage is
```
python experiments/[data]/[student network]/KD_LSHL2_BCE.py
e.g.
python experiments/voc2007/resnet18/KD_LSHL2_BCE.py
```

## Citing this repository

If you find this code useful in your research, please consider citing us:

```
@article{LSHFM,
  title={Distilling knowledge by mimicking features},
  author={Wang, Guo-Hua and Ge, Yifan and Wu, Jianxin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
}
```
