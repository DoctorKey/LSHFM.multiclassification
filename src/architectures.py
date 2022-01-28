import torch
import os
import torch.nn as nn
import torchvision
from functools import partial

from .model.vgg import _vgg_gap_fc
from .model.mobilenet import MobileNetV2
from .model.ResNet_torch import ResNet, Bottleneck, BasicBlock

from .utils import export, parameter_count

@export
def vgg16bn(num_classes=1000, **kwargs):
    model = _vgg_gap_fc('D', True, num_classes, **kwargs)
    return model

@export
def vgg19bn(num_classes=1000, **kwargs):
    model = _vgg_gap_fc('E', True, num_classes, **kwargs)
    return model

@export
def mobilenet_v2(num_classes=1000, **kwargs):
    model = MobileNetV2(num_classes, **kwargs)
    return model

@export
def mobilenet_v2_GMP(num_classes=1000, **kwargs):
    model = MobileNetV2(num_classes, pool='GMP', **kwargs)
    return model

@export
def mobilenet_v2_GWP(num_classes=1000, **kwargs):
    model = MobileNetV2(num_classes, pool='GWP', **kwargs)
    return model

@export
def resnet18(num_classes=1000, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, **kwargs)
    return model

@export
def resnet34(num_classes=1000, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, **kwargs)
    return model

@export
def resnet50(num_classes=1000, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)
    return model

@export
def resnet50_GMP(num_classes=1000, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, pool='GMP', **kwargs)
    return model

@export
def resnet50_GWP(num_classes=1000, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes, pool='GWP', **kwargs)
    return model

@export
def resnet101(num_classes=1000, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, **kwargs)
    return model

@export
def resnet101_GMP(num_classes=1000, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, pool='GMP', **kwargs)
    return model

@export
def resnet101_GWP(num_classes=1000, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, pool='GWP', **kwargs)
    return model


def create_model(model_name, num_classes, detach_para=False, DataParallel=True, **kwargs):
    model_factory = globals()[model_name]
    model_params = dict(num_classes=num_classes, **kwargs)
    model = model_factory(**model_params)
    if DataParallel:
        model = nn.DataParallel(model.cuda())
    else:
        model = model.cuda()

    if detach_para:
        for param in model.parameters():
            param.detach_()
    return model

def load_pretrained(model, pretrained, arch, LOG, DataParallel=True):
    if os.path.isfile(pretrained):
        LOG.info("=> loading pretrained from checkpoint {}".format(pretrained))
        if DataParallel:
            model = nn.DataParallel(model)
        checkpoint = torch.load(pretrained)
        state_dict = checkpoint['state_dict']
        sd_arch = checkpoint['arch']
        if 'moco' in sd_arch:
            if DataParallel:
                replace_str = 'encoder_q.'
            else:
                replace_str = 'module.encoder_q.'
            state_dict.pop('module.encoder_q.fc.0.weight')
            state_dict.pop('module.encoder_q.fc.0.bias')
            state_dict.pop('module.encoder_q.fc.2.weight')
            state_dict.pop('module.encoder_q.fc.2.bias')
            state_dict = {k.replace(replace_str, ''): v for k,v in state_dict.items()}
            ret = model.load_state_dict(state_dict, strict=False)
        elif 'simclr' in sd_arch:
            if DataParallel:
                state_dict = {'module.{}'.format(k): v for k,v in state_dict.items()}        
            ret = model.load_state_dict(state_dict, strict=False)
        else:
            ret = model.load_state_dict(state_dict, strict=False)
        LOG.info("=> loaded pretrained {}".format(pretrained))
        LOG.info("=> not load keys {} ".format(ret))
    elif pretrained.startswith('http'):
        state_dict = torch.hub.load_state_dict_from_url(pretrained, map_location='cpu')
        if 'vgg' in arch:
            state_dict.pop('classifier.6.weight')
            state_dict.pop('classifier.6.bias')
        elif 'mobilenet' in arch:
            state_dict.pop('classifier.1.weight')
            state_dict.pop('classifier.1.bias')
        elif 'densenet' in arch:
            state_dict.pop('classifier.weight')
            state_dict.pop('classifier.bias')
        elif 'fish' in arch:
            state_dict = state_dict['state_dict']
            state_dict.pop('module.fish.fish.9.4.1.weight', None)
            state_dict.pop('module.fish.fish.9.4.1.bias', None)
            state_dict = {k.replace('module.', ''): v for k,v in state_dict.items()}
        elif 'se_res' in arch:
            state_dict.pop('last_linear.weight')
            state_dict.pop('last_linear.bias')
        elif 'efficientnet' in arch:
            state_dict.pop('_fc.weight')
            state_dict.pop('_fc.bias')
        elif 'hrnet' in arch:
            state_dict.pop('classifier.weight')
            state_dict.pop('classifier.bias')
        elif 'ViT' in arch:
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
        else:
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
        ret = model.load_state_dict(state_dict, strict=False)
        if DataParallel:
            model = nn.DataParallel(model)
        LOG.info("=> loaded pretrained {} ".format(pretrained))
        LOG.info("=> not load keys {} ".format(ret))
    else:
        if DataParallel:
            model = nn.DataParallel(model)
        LOG.info("=> NOT load pretrained")
    return model

def fix_BN_stat(module):
    classname = module.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        module.eval()

def fix_BN_learn(module):
    classname = module.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        module.weight.requires_grad_(False)
        module.bias.requires_grad_(False)

