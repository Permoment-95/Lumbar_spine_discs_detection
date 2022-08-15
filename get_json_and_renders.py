import os
from src import config
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= config.NUM_DEVICE
DEVICE='cuda'

import json
import pandas as pd
import torch
import cv2
import numpy as np
import sys
import math
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle

from src.preprocessing import LumbarDataset, get_aug_test
from src.architecture import LumbarHmModel

from src.postprocessing import topk, get_coordinates_hm, annotation_format, intersection_over_union


model = LumbarHmModel(encoder = config.ENCODER, decoder = config.DECODER).to(DEVICE)
model.backbone.load_state_dict(torch.load(config.BEST_MODEL)['model_state_dict'])
model.eval();


def render_and_save_img(img, bboxes, file):
    fig, ax = plt.subplots()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    ax.imshow(img, cmap='gray')
    ax.scatter(bboxes[:,0,0], bboxes[:,0,1], s=100)
    ax.scatter(bboxes[:,1,0], bboxes[:,1,1], s=100)
    ax.scatter(bboxes[:,2,0], bboxes[:,2,1], s=100)
    ax.scatter(bboxes[:,3,0], bboxes[:,3,1], s=100)
    fig.canvas.draw()
    annotated_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    annotated_img = annotated_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    try:
        plt.imsave(os.path.join("../renders/", file.split('.')[0]+'.jpg'), annotated_img)
    except:
        os.mkdir("../renders/")
        plt.imsave(os.path.join("../renders/", file.split('.')[0]+'.jpg'), annotated_img)
    plt.close()
    
path = config.PATH_TO_TEST_IMAGES
output_json = {}

for file in os.listdir(path):
    output_json[file.split('.')[0]] = {}
    abs_path = os.path.join(path, file)
    
    img = np.load(abs_path).astype('float32')
    image_resize = cv2.resize(img, (512, 512))
    image_resize = image_resize / image_resize.max()
    
    out = model(torch.tensor([image_resize.reshape(1,512,512)]).to(DEVICE)).detach().cpu()
    source_shape = img.shape
    
    out_resize = torch.tensor(cv2.resize(out.permute(0,2,3,1)[0].numpy(), (source_shape[1], source_shape[0])).reshape(1,source_shape[0],source_shape[1],5)).permute(0,3,1,2)
    bboxes, cts = get_coordinates_hm(out_resize)
    outputs, bboxes = annotation_format(bboxes)
    
    for i in range(len(outputs)):
        output_json[file.split('.')[0]][i] = outputs[i]
    
    render_and_save_img(img, bboxes, file);

with open('../test_output.pkl', 'wb') as f:
    pickle.dump(output_json, f)    