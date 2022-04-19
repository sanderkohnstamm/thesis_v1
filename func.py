import torch
import torchy
from torch.utils.data import Subset, ConcatDataset
import HSIC
from collections import Counter

import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter


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
            idx = np.cumsum(np.ones(domain_targets.shape[0]))[bools]-1
            data_subset = Subset(datasets[d], idx)
            if len(data_subset) == max_sizes[j]:
                domain_list.append(data_subset)
            else:
                sampled_idx = np.random.choice(idx, size=max_sizes[j])
                domain_list.append(Subset(data_subset, sampled_idx))

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

def train(net, criterion, optimizer, train_loader, epochs=20, gamma=0.5, valid_loader=None, 
                        use_hsic=False, writer=False, min_valid_loss=1000, verbose=False, wb=False):

    if verbose: print(f'Using HSIC:{use_hsic}')
    
    if wb: wandb.watch(net)

    for epoch in range(epochs):  # loop over the dataset multiple times

        train_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data

            optimizer.zero_grad()

            outputs = [net(input)[1] for input in inputs]

            basic_loss = sum([criterion(output, labels) for output in outputs])
            # zero the parameter gradients
            
            if len(inputs)==2: 
                HSIC_loss = HSIC.hsic_normalized(outputs[0], outputs[1])
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

            if writer:
                writer.add_scalar('HSIC', HSIC_loss, epoch*len(train_loader)+i)
                writer.add_scalar('Training loss', basic_loss, epoch*len(train_loader)+i)

            if valid_loader:

                valid_loss = 0.0
                net.eval()     # Optional when not using Model Specific layer
                correct = 0
                total = 0
                for data, labels in valid_loader:
                    if len(data)>1:
                        data1, data2 = data
                        _, val_out1 = net(data1)
                        _, val_out2 = net(data2)
                        _, predicted1 = torch.max(val_out1.data, 1)
                        _, predicted2 = torch.max(val_out2.data, 1)
                        correct += (predicted1 == labels).sum().item() + (predicted2 == labels).sum().item()
                        loss = criterion(val_out1,labels) + criterion(val_out2,labels)

                    else:
                        _, val_out = net(data[0])
                        _, predicted = torch.max(val_out.data, 1)
                        correct += (predicted == labels).sum().item()
                        loss = criterion(val_out,labels)
                        
                    valid_loss = loss.item() * data[0].size(0) 

                    total += labels.size(0)
                    
                    acc = (100 * correct // total)
                # print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(trainloaders[0])} \t\t Validation Loss: {valid_loss / len(validate_loader)}')
                if wb:
                    wandb.log({'Validation loss': valid_loss/len(valid_loader)})
                    wandb.log({'Acc': acc})
                if writer:
                    writer.add_scalar('Validation loss', valid_loss/len(valid_loader), epoch*len(train_loader)+i)
                if min_valid_loss > valid_loss:
                    # print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                    min_valid_loss = valid_loss
                    # Saving State Dict
                    torch.save(net.state_dict(), 'saved_model.pth')   
    
        if verbose: print(train_loss)
    if verbose: print('Finished Training')
    return min_valid_loss
    

def test_model(test_loader, path='saved_model.pth'):
    
    correct = 0
    total = 0

    test_net = torchy.Net()
    test_net.load_state_dict(torch.load(path))

    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for data in test_loader:
            features, labels = data
            # calculate outputs by running images through the network
            _, outputs = test_net(features[0])
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            test_predictions.extend(predicted.detach().cpu().tolist())
            test_targets.extend(labels.detach().cpu().tolist())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

#     print(f'Accuracy of the network on the {len(test_loader)*BATCH_SIZE} test images: {100 * correct // total} %')
    return test_predictions, test_targets, (100 * correct // total)