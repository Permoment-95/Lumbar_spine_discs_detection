import os
import cv2
import sys
import numpy as np

import optuna
import logging
import albumentations as A
import segmentation_models_pytorch as smp
from pathlib import Path
import torch
import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
import catalyst
from catalyst import dl
from catalyst import metrics
from src import config


def randAugment(N, M, p=0.5, mode='default'):#, log_pth='./'):
    M = M-1
    # Magnitude(M) search space  
    if mode == 'default':
        rot = 45
    else:
        rot = np.linspace(0,45,10)
        rot = rot[M]
    shear = np.linspace(0,10,10)
    sig = np.linspace(50,150,10)
    aaff = np.linspace(50,150,10)
    gdist = np.linspace(0.1,0.9,10)
    sola = np.linspace(0.1,0.9,10)
    cont = [np.linspace(-0.8,-0.1,10),np.linspace(0.1,2,10)]
    bright = np.linspace(0.1,0.7,10)
    shar = np.linspace(0.1,0.9,10)
    # Transformation search space
    Aug =[
        #Defaults
        A.OneOf([
          A.Resize(config.RESIZE_SHAPE[0],config.RESIZE_SHAPE[1]),
        ], p=1.),
        A.HorizontalFlip(),
        A.Rotate(limit=rot),
        #Geometrical
        A.Affine(shear=shear[M], p=p),
        A.ElasticTransform(p=p, alpha=2000, sigma=sig[M], alpha_affine=aaff[M]),
        A.GridDistortion(p=p, distort_limit=gdist[M]),
        #Color Based
        A.Solarize(threshold=sola[M], p=p),
        A.RandomBrightnessContrast(contrast_limit=[cont[0][M], cont[1][M]],
                                   brightness_limit=0, p=p),
        A.RandomBrightnessContrast(brightness_limit=bright[M],
                                   contrast_limit=0, p=p),
        A.Sharpen(alpha=shar[M], lightness=shar[M], p=p)]
    
    if mode == 'default':
        ops = Aug[:3]
    else:
        ops = Aug[:3]
        rnd = np.random.choice(Aug[3:], N)
        for rn in rnd:
            ops.append(rn)
    
    transforms = A.Compose(ops)
    return transforms

def get_aug_test(p=1):
    return A.Compose([
        A.Resize(config.RESIZE_SHAPE[0],config.RESIZE_SHAPE[1]),
        #RandomBrightnessContrast(),
        
    ])

class LumbarDataset(Dataset):
    def __init__(self, data, tfms=None, train=True):
        self.data = data
        self.tfms = tfms
        self.train = train
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        
        hm = np.load(self.data.iloc[idx]['path_hm'])
        c,w,h = hm.shape
        img = np.load(self.data.iloc[idx]['path_slice'])
        source_shape = img.shape
        img = img / img.max()
        
        if self.tfms is not None:
            transforms = self.tfms(image=img,mask=np.transpose(hm, (1,2,0)))
            img = transforms['image']
            hm = transforms['mask']
        if self.train:
            return img.reshape(1, config.RESIZE_SHAPE[0],config.RESIZE_SHAPE[1]).astype('float32'), np.transpose(hm, (2,0,1)).astype('float32')
        else:
            return img.reshape(1, config.RESIZE_SHAPE[0],config.RESIZE_SHAPE[1]).astype('float32'), np.transpose(hm, (2,0,1)).astype('float32'), source_shape, self.data.iloc[idx]['path_hm']