import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def mnist():
    # exchange with the corrupted mnist dataset
    train_data_img = torch.tensor([])
    train_data_lbl = torch.tensor([])
    test_data_img = torch.tensor([])
    test_data_lbl = torch.tensor([])

    for i in range(5):
        filepath = 'D:\Downloads\DTU\semester 2\MLOps\dtu_mlops\data\corruptmnist\\train_' + str(i) + '.npz'
        data = np.load(filepath)
        data_img = torch.Tensor(data['images'])
        data_lbl = torch.Tensor(data['labels'])
        data_lbl = data_lbl.type(torch.LongTensor)

        train_data_img = torch.cat((train_data_img, data_img))
        train_data_lbl = torch.cat((train_data_lbl, data_lbl))

    filepath = 'D:\Downloads\DTU\semester 2\MLOps\dtu_mlops\data\corruptmnist\\test.npz'
    data = np.load(filepath)
    data_img = torch.Tensor(data['images'])
    data_lbl = torch.Tensor(data['labels'])
    data_lbl = data_lbl.type(torch.LongTensor)

    test_data_img = torch.cat((test_data_img, data_img))
    test_data_lbl = torch.cat((test_data_lbl, data_lbl))

    train_dataset = TensorDataset(train_data_img,train_data_lbl) # create train dataset
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True) # create train dataloader

    test_dataset = TensorDataset(test_data_img,test_data_lbl) # create test dataset
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True) # create test dataloader
    
    return train_dataloader, test_dataloader
