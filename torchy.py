
import torch
import torch.nn as nn
import torch.nn.functional as F
import thesis.func as func


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
        X = [d[index][0] for d in self.data_list]
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


# from https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/7
class MyCustomResnet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        resnet18 = models.resnet18(pretrained=pretrained)
        # here we get all the modules(layers) before the fc layer at the end
        # note that currently at pytorch 1.0 the named_children() is not supported
        # and using that instead of children() will fail with an error
        self.features = nn.ModuleList(resnet18.children())[:-1]
        # Now we have our layers up to the fc layer, but we are not finished yet 
        # we need to feed these to nn.Sequential() as well, this is needed because,
        # nn.ModuleList doesnt implement forward() 
        # so you cant do sth like self.features(images). Therefore we use 
        # nn.Sequential and since sequential doesnt accept lists, we 
        # unpack all the items and send them like this
        self.features = nn.Sequential(*self.features)
        # now lets add our new layers 
        in_features = resnet18.fc.in_features
        # from now, you can add any kind of layers in any quantity!  
        # Here I'm creating two new layers 
        self.fc0 = nn.Linear(in_features, 256)
        self.fc0_bn = nn.BatchNorm1d(256, eps = 1e-2)
        self.fc1 = nn.Linear(256, 7)
        self.fc1_bn = nn.BatchNorm1d(256, eps = 1e-2)
        
        # initialize all fc layers to xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain = 1)

    def forward(self, input_imgs):
       # now in forward pass, you have the full control, 
       # we can use the feature part from our pretrained model  like this
        output = self.features(input_imgs)

        # since we are using fc layers from now on, we need to flatten the output.
        # we used the avgpooling but we still need to flatten from the shape (batch, 1,1, features)
        # to (batch, features) so we reshape like this. input_imgs.size(0) gives the batchsize, and 
        # we use -1 for inferring the rest
        output = output.view(input_imgs.size(0), -1)
        embs = output
       # and also our new layers. 
        output = self.fc0_bn(F.relu(self.fc0(output)))
        output = self.fc1_bn(F.relu(self.fc1(output)))
                
        return embs, output



def get_feature_loaders(data_dict, batch_size, train_names, valid_name=None, test_name=None, verbose=True):

    data, l = func.dict_to_data(data_dict, train_names) 
    dataset = Dataset(data, l)
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
        val = Dataset(data, l)
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
        test = Dataset(data, l)
    else:
        if verbose: print('No testing')
        test =  None
        train = dataset
    

    datasets = [train, val, test]
    train_loader, valid_loader, test_loader = [DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=0) if d else d for d in datasets]

    return train_loader, valid_loader, test_loader




def get_image_loaders(datasets, batch_size, train_names, valid_name=None, test_name=None, verbose=True):
    
    
    data, l = func.bootstrap(datasets, train_names)
    dataset = Dataset(data, l)
    if verbose: print(f'Training on {train_names}')
    # Get validation
    if valid_name=='split':
        if verbose: print('Validation on trainingset split')
        r = 0.8
        train, val = random_split(dataset, [math.floor(r*len(dataset)), math.ceil((1-r)*len(dataset))])
    elif valid_name:
        if verbose: print(f'Validation on {valid_name}')
        train = dataset
        data, l = func.bootstrap(datasets, valid_name)
        val = Dataset(data, l)
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
        data, l = func.bootstrap(datasets, test_name)
        test = Dataset(data, l)
    else:
        if verbose: print('No testing')
        test =  None
        train = dataset
    
    datasets = [train, val, test]
    train_loader, valid_loader, test_loader = [DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=0) if d else d for d in datasets]



    return train_loader, valid_loader, test_loader
