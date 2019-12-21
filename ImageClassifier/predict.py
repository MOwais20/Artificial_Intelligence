## All the imports here
import argparse
import json
import torch.nn.functional as f
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch import optim
import torch.utils.data 
from torchvision import datasets, transforms, models
from collections import OrderedDict
import utils



parser = argparse.ArgumentParser(description="Image Classification")

parser.add_argument('input', action='store', type=str, default='./flowers/test/1/image_06743.jpg', help="Path to image")
parser.add_argument('checkpoint', default='checkpoint.pth', type=str, help="directory to save checkpoints")
parser.add_argument('--topk', action='store', default='5',help='Top K most likely classes', type=int)
parser.add_argument('--category_names', default='cat_to_name.json', help='category_names', action="store")
parser.add_argument('--gpu', default='gpu', dest='gpu')
 
args = parser.parse_args()

image_path = args.input
path = args.checkpoint
topk = args.topk
device = args.gpu
json_file = args.category_names

def main():
    
    model = utils.loading_checkpoint(path)
    probability = utils.predict(image_path, model, topk, device, json_file)
    
if __name__ == "__main__":
    main()