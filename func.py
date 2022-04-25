import torch
# import thesis.torchy as torchy
from torch.utils.data import Subset, ConcatDataset
import thesis.HSIC as HSIC
from collections import Counter

import numpy as np
import wandb


def bootstrap(datasets, domain_names):
    data_list = []

    classes = set(datasets[domain_names[0]].targets)
    num_classes = len(set(datasets[domain_names[0]].targets))
    sizes = np.zeros((len(domain_names), num_classes))

    for i, domain_key in enumerate(domain_names):
        for j, class_size in enumerate(list(Counter(datasets[domain_key].targets).values())):
            sizes[i,j] = class_size  
    
    max_sizes = np.max(sizes, axis=0).astype(int)
    labels = torch.from_numpy(np.repeat(np.arange(num_classes), max_sizes))

    for d in domain_names:
        domain_list = []
        domain_targets = np.array(datasets[d].targets)
        for j, class_key in enumerate(classes):
            bools = domain_targets==class_key
            idx = (np.cumsum(np.ones(domain_targets.shape[0]))[bools]-1).astype(int)
            data_subset = Subset(datasets[d], idx)
            if len(data_subset) == max_sizes[j]:
                domain_list.append(data_subset)
            else:
                sampled_idx = np.random.choice(idx, size=max_sizes[j])
                domain_list.append(Subset(datasets[d], sampled_idx))

        data_list.append(ConcatDataset(domain_list))
        
    return data_list, labels        


def dict_to_data(feature_dict, domain_names):
    data_list = []

    num_classes = len(feature_dict[domain_names[0]])
    sizes = np.zeros((len(domain_names), num_classes))

    for i, domain_key in enumerate(domain_names):
        for j, class_key in enumerate(feature_dict[domain_key].keys()):
            sizes[i,j] =  len(feature_dict[domain_key][class_key])            
    
    max_sizes = np.max(sizes, axis=0).astype(int)
    labels = torch.from_numpy(np.repeat(np.arange(num_classes), max_sizes))
    
    for i, domain_key in enumerate(domain_names):
        domain_list = []
        for j, class_key in enumerate(feature_dict[domain_key].keys()):
            data = feature_dict[domain_key][class_key]
            data= torch.stack(data).squeeze()
            if len(data) == max_sizes[j]:
                domain_list.append(data)
            else:
                idx = np.random.randint(data.shape[0], size=max_sizes[j])
                domain_list.append(data[idx])
        
        data_list.append(torch.vstack(domain_list))
        
    return data_list, labels


def train(net, criterion, optimizer, train_loader, epochs=20, gamma=0.5,  device='cpu', valid_loader=None, 
                        use_hsic=False, writer=False, min_valid_loss=1000, verbose=False, wb=False):


    net.to(device)
    if verbose: 
        print(f'Using HSIC:{use_hsic}')
        print(f'Cuda:{device}')
    if wb: wandb.watch(net)

    for epoch in range(epochs):  # loop over the dataset multiple times

        if verbose: print(f'Epoch {epoch}')

        train_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data
            inputs, labels = inputs, labels.to(device)

            optimizer.zero_grad()

            outputs = [net(i.to(device)) for i in inputs]

            basic_loss = sum([criterion(output, labels) for _, output in outputs])
            # zero the parameter gradients
            
            if len(inputs)==2: 
                emb1, emb2 = [emb for emb, _ in outputs]
                HSIC_loss = HSIC.hsic_normalized(emb1, emb2)
            else: 
                HSIC_loss = np.NaN

            if use_hsic:
                if HSIC_loss == HSIC_loss:
                    loss = basic_loss + gamma*(1-HSIC_loss)
                    
                else:
                    if verbose: print('out of bounds, epoch/i: ', epoch,'/', i)
                    loss = basic_loss
            else:
                loss = basic_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += basic_loss.item()
            if wb:
                wandb.log({"HSIC": HSIC_loss})
                wandb.log({"Training loss": basic_loss})

            if verbose:
                print("HSIC:", HSIC_loss)
                print("Training loss:", basic_loss)

            if valid_loader:

                valid_loss = 0.0
                net.eval()     # Optional when not using Model Specific layer
                correct = 0
                total = 0
                for data, labels in valid_loader:
                    data, labels = data, labels.to(device)

                    data = [d.to(device) for d in data]

                    val_outputs = [net(d) for d in data]
                    predicted = [torch.max(val_out.data, 1) for _, val_out in val_outputs]

                    correct = sum([(pred.indices == labels).sum().item() for pred in predicted])
                    loss = sum([criterion(val_out,labels) for _, val_out in val_outputs])

                        
                    valid_loss = loss.item() * data[0].size(0) 

                    total += labels.size(0)
                    
                    acc = (100 * correct // total)

                if wb:
                    wandb.log({'Validation loss': valid_loss/len(valid_loader)})
                    wandb.log({'Validation accuracy': acc})
                if min_valid_loss > valid_loss:
                    if verbose: print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                    min_valid_loss = valid_loss
                    # Saving State Dict
                    torch.save(net.state_dict(), 'saved_model.pth')   
    
        if verbose: print(train_loss)
    if verbose: print('Finished Training')
    return min_valid_loss
    

def test_model(test_net, test_loader, path='saved_model.pth', verbose=False):
    
    correct = 0
    total = 0

    # test_net = torchy.Net()
    test_net.load_state_dict(torch.load(path))

    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for data, labels in test_loader:
            
            # calculate outputs by running images through the network
            outputs = test_net(data)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs[1].data, 1)

            test_predictions.extend(predicted.detach().cpu().tolist())
            test_targets.extend(labels.detach().cpu().tolist())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if verbose: print(f'Accuracy of the network on the {len(correct)} test images: {100 * correct // total} %')

    return test_predictions, test_targets, (100 * correct // total)