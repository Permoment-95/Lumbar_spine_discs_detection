import os
from src import config
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=config.NUM_DEVICE
DEVICE='cuda'

import os
import cv2
import sys
import numpy as np
import glob
import optuna
import logging
import albumentations as A
import segmentation_models_pytorch as smp
from datetime import datetime
from pathlib import Path
import torch
import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
from catalyst import dl
from catalyst import metrics
import pandas as pd
import catalyst

import pydicom as dicom
import albumentations as A

from sklearn.model_selection import train_test_split


data = pd.read_csv(config.PATH_TO_DATA)
train_df, val_df = train_test_split(data, test_size=0.2, random_state=config.SEED)

def randAugment(N, M, p=0.5, mode='default'):
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
        
    ])

class LumbarDataset(Dataset):
    def __init__(self, data, tfms=None):
        self.data = data
        self.tfms = tfms
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
        return img.reshape(1, config.RESIZE_SHAPE[0],config.RESIZE_SHAPE[1]).astype('float32'), np.transpose(hm, (2,0,1)).astype('float32')


def model_setup(encoder_name, decoder_name):
    encoder_weights = 'noisy-student' if encoder_name.startswith('timm-efficientnet') else 'imagenet'
    
    model = getattr(smp, decoder_name)(
                                       encoder_name,
                                       in_channels=1,
                                       activation='sigmoid',
                                       classes=5, 
                                       encoder_weights=encoder_weights,
    )

    return model

def objective(trial):
    global criterion_name
    
    N = trial.suggest_int("N", 0, 7)
    M = trial.suggest_int("M", 1, 10)

    encoder = trial.suggest_categorical("encoder", ["timm-efficientnet-b2", "resnet34", "resnext50_32x4d",'vgg16', 'se_resnet50',"timm-efficientnet-b4","timm-efficientnet-b3", "timm-efficientnet-b5"])
    decoder = trial.suggest_categorical("decoder", ["MAnet", "Linknet", "Unet", "UnetPlusPlus"])
    model = model_setup(encoder, decoder).cuda()
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True) 
    criterion = nn.MSELoss()
    
        
    dataset_train = LumbarDataset(train_df, randAugment(N, M, 'aug'))
    dataloader_train = DataLoader(dataset_train, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True, num_workers=8)
    
    dataset_val = LumbarDataset(val_df, get_aug_test())
    dataloader_val = DataLoader(dataset_val, batch_size=config.BATCH_SIZE_VAL, shuffle=False, num_workers=8)
    
    loaders = {
    "train": dataloader_train,
    "valid": dataloader_val,
}
    
    runner = dl.SupervisedRunner(
    input_key="features", output_key="logits", target_key="targets", loss_key="loss"
)
    print(f'Runing: {encoder}-{decoder}-MSELoss-lr:{str(lr)}')
    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=config.NUM_EPOCHS,
        logdir=Path(config.PATH_TO_EXPERIMENTS) / f"{encoder}-{decoder}-N{str(N)}-M{str(M)}-MSELoss-lr{str(lr)}-perbatch",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
        callbacks=[
            dl.EarlyStoppingCallback(patience=config.PATINENCE, loader_key='valid', metric_key='loss', minimize=True),
            dl.OptunaPruningCallback(loader_key="valid", metric_key="loss", minimize=True, trial=trial),
        ],
    )
    return trial.best_score


pruner = optuna.pruners.MedianPruner(n_startup_trials=config.N_STARTUP_TRIALS, n_warmup_steps=config.N_WARMUP_STEPS)

study = optuna.create_study(study_name='optuna_tuning_multiclass', 
                               direction="minimize", pruner=pruner, load_if_exists=True)
study.optimize(objective, n_trials=config.N_TRIALS, gc_after_trial=True)
