from __future__ import print_function, division
from tabnanny import verbose

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset, ChainDataset, DataLoader, random_split
import torch.nn.functional as F

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


cudnn.benchmark = True

import pandas as pd
import pickle
import math
import random

import HSIC
import matplotlib.pyplot as plt
import numpy as np
import func
import torchy

import wandb

from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter



if __name__=='__main__':
    data_root = "../../Data/PACS/"


    all_domain_names = ['photo', 'art_painting', 'cartoon', 'sketch']
    domain_names = ['photo', 'art_painting', 'cartoon']

    # means and standard deviations ImageNet because the network is pretrained
    means, stds = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    # Define transforms to apply to each image
    transf = transforms.Compose([ #transforms.Resize(227),      # Resizes short size of the PIL image to 256
                                transforms.CenterCrop(224),  # Crops a central square patch of the image 224 because torchvision's AlexNet needs a 224x224 input!
                                transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                                transforms.Normalize(means,stds) # Normalizes tensor with mean and standard deviation
    ])

    datasets = {}

    for name in os.listdir(data_root):
    
        if not name[0] == '.':
            dataset = torchvision.datasets.ImageFolder(data_root+name, transform=transf)

            datasets[name] = dataset
            print(f"Added :{name}, length: {len(datasets[name])}")
    
    # print(dict(Counter(dataset.targets)))
    num_classes = 7      # 7 classes for each domain: 'dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'
    classes_names = ['Dog', 'Elephant', 'Giraffe', 'Guitar', 'Horse', 'House', 'Person']
    domain_mapping = {'photo':0, 'art_painting':1, 'cartoon':2, 'sketch':3}
    model_path = 'saved_model.pth'

    criterion = nn.CrossEntropyLoss()



    train_names = ['photo', 'art_painting']
    valid_name = ['cartoon']
    test_name = 'split'

    lr=0.001
    use_hsic = True
    batch_size = 128
    epochs = 30
    for gamma in [2.5]:
        print('Lambda: ', gamma)
        for i in range(20):
            wandb.init(project="test-one",
                        entity="skohnie",
                        name=f'{gamma}/{i}',
                        config = {"learning_rate": lr,
                                    "epochs": 100,
                                    "batch_size": 128,
                                    "gamma": gamma,
                                    "Train names": train_names,
                                    "Valid name": valid_name,
                                    "Test name": test_name
                                    }
                    )



            train_loader, valid_loader, test_loader  = torchy.get_image_loaders(datasets, 
                                                                                batch_size,
                                                                                train_names,
                                                                                valid_name,
                                                                                test_name,
                                                                                verbose=True)


            resnet18 = models.resnet18(pretrained=True)
            resnet18.fc1 = nn.Linear(512, 7)

            min_valid_loss = 1000



            optimizer = optim.Adam(resnet18.parameters(), lr=lr)
            min_valid_loss = func.train(resnet18, criterion, optimizer, 
                                        train_loader,
                                        valid_loader=valid_loader,
                                        epochs=30,
                                        use_hsic=use_hsic,
                                        gamma=gamma,
                                        writer=None,
                                        min_valid_loss = min_valid_loss,
                                        wb=True,
                                        verbose=True)

            acc = func.test_model(test_loader, 'saved_model.pth')
            wandb.summary['Test Accuracy'] = acc
            wandb.finish()
