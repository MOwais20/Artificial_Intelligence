# Imports here
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import torch
from torch import nn
from torch import optim
import torch.utils.data 
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import utils



parser = argparse.ArgumentParser(description="train.py")

parser.add_argument('data_dir', type=str, default="./flowers/", help="Requires data directory")
parser.add_argument('--save_dir', default='./checkpoint.pth', type=str, help="directory to save checkpoints")
parser.add_argument('--arch', default='alexnet',help='Architecture')
parser.add_argument('--learning_rate', default=0.001, help='learning rate', action="store", type=float)
parser.add_argument('--hidden_units', default=2048, help='number of neurons in hidden layer', type=int)
parser.add_argument('--dropout', default=0.3, type=float, help='dropout probability')
parser.add_argument('--epochs', default=6, type=int, help='number of epochs for training')
parser.add_argument('--gpu', default='gpu', help='GPU to be used for training?')
 
args = parser.parse_args()
data = args.data_dir
path = args.save_dir
lr = args.learning_rate
epochs = args.epochs
arch = args.arch
dropout = args.dropout
hidden_units = args.hidden_units
device = args.gpu


def main():
    
    train_dataloaders, valid_dataloaders, test_dataloaders = utils.loader(data)
    model, optimizer, criterion = utils.network(arch, dropout, hidden_units, lr, device)
    utils.Train(optimizer, criterion, model, epochs, device)
#    utils.testing(model, test_dataloaders, device)
    utils.save_checkpoint(model, path, arch, hidden_units, dropout, lr)


if __name__== "__main__":
    main()