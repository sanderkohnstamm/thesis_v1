
import torch
import torch.nn as nn
import torch.nn.functional as F
import func
import random
import torchy
import math
from torch.utils.data import DataLoader, random_split

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_list, labels):
        'Initialization'
        self.labels = labels
        self.data_list = data_list

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample and label
        X = [d[index] for d in self.data_list]
        y = self.labels[index]

        return X, y

class AddDomain(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dataset, domain):
        'Initialization'
        self.dataset = dataset
        self.domain = domain

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample and label
        X, y = self.dataset[index]
        d = self.domain

        return X, y, d


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(512, 128)  # 5*5 from image dimension
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 7)

    def forward(self, x):
        emb = F.relu(self.fc1(x))
        emb = F.relu(self.fc2(emb))
        x = self.fc3(emb)
        return emb, x

def get_feature_loaders(data_dict, batch_size, train_names, valid_name=None, test_name=None, verbose=True):

    data, l = func.dict_to_data(data_dict, train_names) 
    dataset = torchy.Dataset(data, l)
    if verbose: print(f'Training on {train_names}')
    # Get validation
    if valid_name=='split':
        if verbose: print('Validation on trainingset split')
        r = 0.8
        train, val = random_split(dataset, [math.floor(r*len(dataset)), math.ceil((1-r)*len(dataset))])
    elif valid_name:
        if verbose: print(f'Validation on {valid_name}')
        train = dataset
        data, l = func.dict_to_data(data_dict, valid_name) 
        val = torchy.Dataset(data, l)
    else:
        if verbose: print('No validation')
        val =  None
        train = dataset
    
    #Get testing
    if test_name=='split':
        if verbose: print('Testing on validation set split')
        r = 0.5
        val, test = random_split(val, [math.floor(r*len(val)), math.ceil((1-r)*len(val))])
    elif test_name:
        if verbose: print(f'Testing on {test_name}')
        data, l = func.dict_to_data(data_dict, test_name) 
        test = torchy.Dataset(data, l)
    else:
        if verbose: print('No testing')
        test =  None
        train = dataset
    

    datasets = [train, val, test]
    train_loader, valid_loader, test_loader = [DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=0) if d else d for d in datasets]



    return train_loader, valid_loader, test_loader




def get_image_loaders(datasets, batch_size, train_names, hsic_name=None, valid_name=None, test_name=None, verbose=True):
    
    
    dataset = torch.utils.data.ConcatDataset([datasets[d] for d in train_names])

    # Get validation
    if valid_name=='split':
        if verbose: print('Validation on trainingset split')
        r = 0.8
        train, val = random_split(dataset, [math.floor(r*len(dataset)), math.ceil((1-r)*len(dataset))])
    elif valid_name:
        if verbose: print(f'Validation on {valid_name}')
        train = dataset
        val = datasets[valid_name]
    else:
        if verbose: print('No validation')
        val =  None
        train = dataset
    
    #Get testing
    if test_name=='split':
        if verbose: print('Testing on validation set split')
        r = 0.5
        val, test = random_split(val, [math.floor(r*len(val)), math.ceil((1-r)*len(val))])
    elif test_name:
        if verbose: print(f'Testing on {test_name}')
        test = datasets[test_name]
    else:
        if verbose: print('No testing')
        test =  None
        train = dataset
    
    #Get HSIC Domain
    if hsic_name:
        if verbose: print(f'HSIC domains:{train_names[0], hsic_name}')
        hsic = datasets[hsic_name]
    else:
        if verbose: print(f'Training on {train_names}, no HSIC')
        hsic = None

    datasets = [train, val, test, hsic]
    train_loader, valid_loader, test_loader, HSIC_loader = [DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=0) if d else d for d in datasets]



    return train_loader, valid_loader, test_loader, HSIC_loader
