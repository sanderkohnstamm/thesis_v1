from __future__ import print_function, division

import numpy as np


import pandas as pd
import pickle

import matplotlib.pyplot as plt
import numpy as np
import func
import torchy






if __name__=='__main__':
    a_file = open("PACS.pkl", "rb")
    feature_dict = pickle.load(a_file)
    num_classes = 7      # 7 classes for each domain: 'dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'
    classes_names = ['Dog', 'Elephant', 'Giraffe', 'Guitar', 'Horse', 'House', 'Person']
    domain_mapping = {'photo':0, 'art_painting':1, 'cartoon':2, 'sketch':3}
    batch_size = 64
    PATH = 'saved_model.pth'
    all_domain_names = ['photo', 'art_painting', 'cartoon', 'sketch']
    domain_names = ['photo', 'art_painting', 'cartoon']

    criterion = nn.CrossEntropyLoss()

    gamma = 1
    i = 0
    lr=0.001


    big_acc = {}
    for i in range(len(all_domain_names)):
        big_acc[all_domain_names[i]] = {}
        for j in range(len(all_domain_names)):
            
            if i==j:
                continue
            train_names = [all_domain_names[i]]
            valid_name = 'split'
            test_name = all_domain_names[j]
            
            big_acc[train_names[0]][test_name] = []


            for run in range(5):

            
                wandb.init(project="Visualise",
                            entity="skohnie",
                            name=f'{train_names[0]}_{test_name}_{i}',
                            config = {"learning_rate": lr,
                                        "epochs": 100,
                                        "batch_size": 128,
                                        "Train names": train_names,
                                        "Valid name": valid_name,
                                        "Test name": test_name
                                        }
                        )
                train_names = [domain_names[i]]
                valid_name = 'split'
                test_name = [domain_names[j]]


                train_loader, valid_loader, test_loader  = torchy.get_feature_loaders(feature_dict, 
                                                                                    domain_mapping,
                                                                                    batch_size,
                                                                                    train_names,
                                                                                    hsic_name,
                                                                                    valid_name,
                                                                                    test_name)

                acc_list = []                          
                min_valid_loss = 1000
            
                net = torchy.Net()
                
                optimizer = optim.Adam(net.parameters(), lr=lr)
                min_valid_loss = func.train(net, criterion, optimizer, train_loader,
                                                valid_loader=valid_loader,
                                                epochs=20,
                                                hsic_loader=hsic_loader,
                                                gamma=gamma,
                                                writer=writer,
                                                min_valid_loss = min_valid_loss)


                if test_name:
                    acc = func.test_model(test_loader, 'saved_model.pth')
                    big_acc[all_domain_names[i]][all_domain_names[j]].append(acc)
            avg_acc = sum(acc_list) / len(acc_list)
            print(f'{i} Training: {train_names}, Valid: {valid_name}, Testing: {test_name} = {avg_acc}%')
            i+=1
            wandb.summary['Test Accuracy'] = avg_acc
            wandb.finish()

                