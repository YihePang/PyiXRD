import torch
import numpy as np
import os
import json
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter

def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        for row in file.readlines():
            ids_labels = json.loads(row)
    return ids_labels

def data_add_angle(features,angles):
    return np.stack((features, angles), axis=0)

def get_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

def load_dataset(directory,ids_labels_path,data_length, len_slice,flag='crystal system'):
    all_files=get_all_files(directory)
    ids_labels=read_json(path=ids_labels_path)
    temp=list(ids_labels.keys())
    result = [item for item in all_files if any(os.path.basename(item).startswith(sub + '.') for sub in temp)]
    angle = [round(0.01 * i + 5.0, 2) for i in range(8501)]
    angles = angle[::len_slice[data_length]]
    data=[]
    labels=[]
    for file in result:
        d = np.load(file, allow_pickle=True).item()
        feature = d.get('data')
        feature=feature[::len_slice[data_length]]
        sequences=data_add_angle(feature,angles)
        data.append(sequences)
        label = d.get(flag)
        labels.append(int(label))
    data=np.array(data)
    print('MP shape:',data.shape)
    counter_result = Counter(labels)
    print('MP Class count:',counter_result)
    labels=np.array(labels)
    return data,labels

def load_xrd_dataset(batch_size,path,directory,data_length, len_slice,flag='crystal system'):#path=['./split_data/train.json','./split_data/val.json','./split_data/test.json']
    train_data, train_labels=load_dataset(directory=directory,ids_labels_path=path[0], data_length=data_length,len_slice=len_slice,flag=flag)
    val_data, val_labels=load_dataset(directory=directory,ids_labels_path=path[1], data_length=data_length,len_slice=len_slice,flag=flag)
    test_data, test_labels=load_dataset(directory=directory,ids_labels_path=path[2], data_length=data_length,len_slice=len_slice,flag=flag)

    x_train = torch.tensor(train_data, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    x_val = torch.tensor(val_data, dtype=torch.float32)
    y_val = torch.tensor(val_labels, dtype=torch.long)
    x_test = torch.tensor(test_data, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return y_train,train_loader,val_loader,test_loader
