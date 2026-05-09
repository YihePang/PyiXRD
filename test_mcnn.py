"""
该代码仅可用于对XRD测试集进行测试，完整模型训练/验证/测试代码见mcnn.py
This code can only be used to test the XRD test set. See mcnn.py for the complete model training/verification/test code.
"""
import json
import os
import random
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
import torch.nn.functional as F
from evaluation_index import *

class BiDepthConv_add(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1,3), stride=(1,1), padding=(0,1)):
        super(BiDepthConv_add, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x

        out_forward = self.depthwise(x)
        out_forward = self.pointwise(out_forward)

        x_flipped = torch.flip(x, dims=[3])
        out_backward = self.depthwise(x_flipped)
        out_backward = self.pointwise(out_backward)
        out_backward = torch.flip(out_backward, dims=[3])

        out=out_forward+out_backward
        atten=identity+out
        return atten
    
class MultiScaleCNN(nn.Module):
    def __init__(self, classes_num, n_channel):
        super(MultiScaleCNN, self).__init__()
        # Shared initial convolution block
        self.shared_conv = nn.Sequential(
            nn.Conv2d(n_channel, 32, kernel_size=(1, 25), stride=(1, 1), padding=(0, 12)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        )

        # Branch 1: smaller kernel size
        self.branch1 = nn.Sequential(
            BiDepthConv_add(32,32),
            nn.Conv2d(32, 64, kernel_size=(1, 32), stride=(1, 1), padding=(0, 16)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            BiDepthConv_add(64,64),
            nn.Conv2d(64, 128, kernel_size=(1, 32), stride=(1, 1), padding=(0, 16)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            BiDepthConv_add(128,128),
            nn.Conv2d(128, 256, kernel_size=(1, 32), stride=(1, 1), padding=(0, 16)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            BiDepthConv_add(256,256),
            nn.Conv2d(256, 512, kernel_size=(1, 32), stride=(1, 1), padding=(0, 16)),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AdaptiveMaxPool2d((1, 1))
        )

        # Branch 2: medium kernel size
        self.branch2 = nn.Sequential(
            BiDepthConv_add(32,32),
            nn.Conv2d(32, 64, kernel_size=(1, 64), stride=(1, 1), padding=(0, 32)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            BiDepthConv_add(64,64),
            nn.Conv2d(64, 128, kernel_size=(1, 64), stride=(1, 1), padding=(0, 32)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            BiDepthConv_add(128,128),
            nn.Conv2d(128, 256, kernel_size=(1, 64), stride=(1, 1), padding=(0, 32)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            BiDepthConv_add(256,256),
            nn.Conv2d(256, 512, kernel_size=(1, 64), stride=(1, 1), padding=(0, 32)),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AdaptiveMaxPool2d((1, 1))
        )

        # Branch 3: larger kernel size
        self.branch3 = nn.Sequential(
            BiDepthConv_add(32,32),
            nn.Conv2d(32, 64, kernel_size=(1, 128), stride=(1, 1), padding=(0, 64)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            BiDepthConv_add(64,64),
            nn.Conv2d(64, 128, kernel_size=(1, 128), stride=(1, 1), padding=(0, 64)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            BiDepthConv_add(128,128),
            nn.Conv2d(128, 256, kernel_size=(1, 128), stride=(1, 1), padding=(0, 64)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            BiDepthConv_add(256,256),
            nn.Conv2d(256, 512, kernel_size=(1, 128), stride=(1, 1), padding=(0, 64)),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AdaptiveMaxPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 3, 512), 
            nn.ReLU(),
            nn.Linear(512, classes_num)  
        )

    def forward(self, x):
        x = x.unsqueeze(2)  
        x = self.shared_conv(x)
        branch1_out = self.branch1(x).flatten(start_dim=1)
        branch2_out = self.branch2(x).flatten(start_dim=1)
        branch3_out = self.branch3(x).flatten(start_dim=1)
        concat_out = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)  
        output = self.fc(concat_out)
        return output

def get_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

def read_json(path):#'./split_data/test.json'
    with open(path, 'r', encoding='utf-8') as file:
        for row in file.readlines():
            ids_labels = json.loads(row)
    return ids_labels

def data_add_angle(features,angles):
    return np.stack((features, angles), axis=0)

def load_dataset(directory,test_path,data_length, len_slice,batch_size, flag='crystal system'):#no ids
    all_files=get_all_files(directory)
    ids_labels=read_json(path=test_path)
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

    test_data = torch.tensor(data, dtype=torch.float32)
    test_labels = torch.tensor(labels, dtype=torch.long)
    test_dataset=TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def test(model, test_loader, flag, device):
    model.eval()
    correct = 0
    total = 0
    top_correct0 = 0
    top_correct1 = 0
    true_labels=[]
    pred_labels=[]
    all_predicted = []
    all_labels0 = []
    average_flag = ['micro', 'macro', 'weighted', None]
    all_probs = []
    all_labels1 = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    if flag == 'crystal system':
        k1 = 2
        k2 = 3
    elif flag == 'space group':
        k1 = 3
        k2 = 5
    k_max = max(k1, k2)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) 
            _, predicted = torch.max(outputs, 1)
            true_labels.append(labels)
            pred_labels.append(predicted)

            _, topk_indices = torch.topk(outputs, k_max, dim=1)
            top_preds0 = topk_indices[:, :k1]
            top_preds1 = topk_indices[:, :k2]

            labels_view = labels.view(-1, 1)
            top_correct0 += (top_preds0 == labels_view).any(dim=1).sum().item()
            top_correct1 += (top_preds1 == labels_view).any(dim=1).sum().item()

            del topk_indices, top_preds0, top_preds1, labels_view

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predicted_cpu = predicted.cpu()
            labels_cpu = labels.cpu()

            for label, prediction in zip(labels_cpu, all_predicted_cpu):  
                class_correct[label.item()] += (prediction == label).item()
                class_total[label.item()] += 1

            all_predicted.extend(all_predicted_cpu.tolist())
            all_labels0.extend(labels_cpu.tolist())

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_labels1.extend(labels_cpu.numpy())

            del outputs, predicted, probs, labels_cpu, all_predicted_cpu

    acc = accuracy(correct, total)
    top_acc0 = top_correct0 / total if total > 0 else 0
    top_acc1 = top_correct1 / total if total > 0 else 0
    macro_pre = precision(average_flag[1], all_predicted, all_labels0)
    macro_f1 = F1(average_flag[1], all_labels0, all_predicted)
    pr_auc_macro = macro_pr_auc(all_probs, all_labels1)

    index = [acc, top_acc0, top_acc1, macro_pre, macro_f1, pr_auc_macro]
    return index

if __name__ == '__main__':
    batch_size=64
    flag='crystal system' # or 'space group'
    data_length = 8501
    len_slice = {8501: 1}
    classes = {'crystal system': 7, 'space group': 230}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    test_path=r'./split_data/test.json'
    all_files_path=r'./XRD_patterns/'
    model_path={'crystal system':'./model_pth/crystal_system.pth', 'space group':'./model_pth/space_group.pth'}

    model = MultiScaleCNN(classes_num=classes[flag], n_channel=2).to(device)
    checkpoint = torch.load(model_path[flag])
    # checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    test_loader=load_dataset(all_files_path,test_path,data_length, len_slice,batch_size, flag=flag)
    index=test(model, test_loader, flag, device)

    name = ['Accuracy', 'top_accuracy0', 'top_accuracy1', 'Macro Precision', 'Macro F1', 'PR-AUC(macro)']
    print(name)
    print(index)


