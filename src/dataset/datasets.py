import socket
import torch
import torchvision
import torchvision.transforms as transforms

from .COCO import CocoDetection
from .VOC import VOC2007

from ..utils import export

@export
def COCO2014(args=None):
    # TODO: edit below according your system
    train_pic = '/opt/Dataset/COCO2014/train2014'
    train_anno = '/opt/Dataset/COCO2014/annotations/instances_train2014.json'
    val_pic = '/opt/Dataset/COCO2014/val2014'
    val_anno = '/opt/Dataset/COCO2014/annotations/instances_val2014.json'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transformation = transforms.Compose([
                                    transforms.Resize((args.input_size, args.input_size)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
                                    transforms.ToTensor(),
                                    normalize,
                                ])
    eval_transformation = transforms.Compose([
                                    transforms.Resize((args.input_size, args.input_size)),
                                    transforms.ToTensor(),
                                    normalize,
                                ])

    train_dataset = CocoDetection(train_pic, train_anno, train_transformation)
    val_dataset = CocoDetection(val_pic, val_anno, eval_transformation)

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'num_classes': 80
    }

@export
def voc2007(args=None):
    # TODO: edit below according your system
    data = '/opt/Dataset/VOC'
    # |- VOCdevkit
    # |- VOCtest_06-Nov-2007.tar  
    # |- VOCtrainval_06-Nov-2007.tar

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transformation = transforms.Compose([
                                    transforms.Resize((args.input_size, args.input_size)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
                                    transforms.ToTensor(),
                                    normalize,
                                ])
    eval_transformation = transforms.Compose([
                                    transforms.Resize((args.input_size, args.input_size)),
                                    transforms.ToTensor(),
                                    normalize,
                                ])
    
    def target_transform(target):
        return (target >= 0).float()

    train_dataset = VOC2007(data, 'trainval', transform=train_transformation, target_transform=target_transform)
    val_dataset = VOC2007(data, 'test', transform=eval_transformation)

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'num_classes': 20
    }



def get_dataset_config(dataset_name, args=None):
    dataset_factory = globals()[dataset_name]
    params = dict(args=args)
    dataset_config = dataset_factory(**params)
    return dataset_config
    
