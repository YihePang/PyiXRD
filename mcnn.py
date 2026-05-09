import torch.nn as nn
import torch
import datetime
from loaddata import load_xrd_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from collections import defaultdict
from evaluation_index import *
from Loss import LabelSmoothingLoss

class BiDepthConv_add(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)):
        super(BiDepthConv_add, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
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

        out = out_forward + out_backward
        atten = identity + out
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
            BiDepthConv_add(32, 32),
            nn.Conv2d(32, 64, kernel_size=(1, 32), stride=(1, 1), padding=(0, 16)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            BiDepthConv_add(64, 64),
            nn.Conv2d(64, 128, kernel_size=(1, 32), stride=(1, 1), padding=(0, 16)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            BiDepthConv_add(128, 128),
            nn.Conv2d(128, 256, kernel_size=(1, 32), stride=(1, 1), padding=(0, 16)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            BiDepthConv_add(256, 256),
            nn.Conv2d(256, 512, kernel_size=(1, 32), stride=(1, 1), padding=(0, 16)),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AdaptiveMaxPool2d((1, 1))
        )

        # Branch 2: medium kernel size
        self.branch2 = nn.Sequential(
            BiDepthConv_add(32, 32),
            nn.Conv2d(32, 64, kernel_size=(1, 64), stride=(1, 1), padding=(0, 32)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            BiDepthConv_add(64, 64),
            nn.Conv2d(64, 128, kernel_size=(1, 64), stride=(1, 1), padding=(0, 32)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            BiDepthConv_add(128, 128),
            nn.Conv2d(128, 256, kernel_size=(1, 64), stride=(1, 1), padding=(0, 32)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            BiDepthConv_add(256, 256),
            nn.Conv2d(256, 512, kernel_size=(1, 64), stride=(1, 1), padding=(0, 32)),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AdaptiveMaxPool2d((1, 1))
        )

        # Branch 3: larger kernel size
        self.branch3 = nn.Sequential(
            BiDepthConv_add(32, 32),
            nn.Conv2d(32, 64, kernel_size=(1, 128), stride=(1, 1), padding=(0, 64)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),

            BiDepthConv_add(64, 64),
            nn.Conv2d(64, 128, kernel_size=(1, 128), stride=(1, 1), padding=(0, 64)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            BiDepthConv_add(128, 128),
            nn.Conv2d(128, 256, kernel_size=(1, 128), stride=(1, 1), padding=(0, 64)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            BiDepthConv_add(256, 256),
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

def train(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    torch.cuda.empty_cache()
    return running_loss / len(train_loader)

def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_labels0 = []
    average_flag = ['micro', 'macro', 'weighted', None]
    all_probs = []
    all_labels1 = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            # for acc
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # for precision f1
            all_predicted.extend(predicted.cpu().tolist())
            all_labels0.extend(labels.cpu().tolist())
            # for PR-AUC
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_labels1.extend(labels.cpu().numpy())
            # for class acc
            for label, prediction in zip(labels, predicted):
                class_correct[label.item()] += (prediction == label).item()
                class_total[label.item()] += 1

    loss = running_loss / len(test_loader)
    class_accuracy = class_acc(class_correct, class_total)
    acc = accuracy(correct, total)
    macro_pre = precision(average_flag[1], all_predicted, all_labels0)
    macro_f1 = F1(average_flag[1], all_labels0, all_predicted)
    pr_auc_macro = macro_pr_auc(all_probs, all_labels1)

    index = {}
    name = ['accuracy', 'macro_precision', 'macro_f1', 'PR_AUC_macro']
    all_index = [acc, macro_pre, macro_f1, pr_auc_macro]
    for i in range(len(name)):
        index[name[i]] = all_index[i]

    return loss, class_accuracy, index

def draw_loss(loss_data1,loss_data2,name1='Training Loss',name2='Validation Loss',save_name="loss_curve.png"):
    epochs = range(1, len(loss_data1) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss_data1, label=name1)
    plt.plot(epochs, loss_data2, label=name2)
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()

def dataset(batch_size, directory,ids_labels_path,data_length, len_slice,flag):
    y_train, train_loader, val_loader, test_loader = load_xrd_dataset(batch_size,ids_labels_path,directory,data_length, len_slice, flag)
    return y_train,train_loader,val_loader,test_loader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    directory = r'./XRD_patterns/'
    ids_labels_path = [r'./split_data/train.json',
                       r'./split_data/val.json',
                       r'./split_data/test.json']
    save_path = r'./result_files/'

    flag = 'crystal system'  # or 'space_group'
    classes = {'crystal system': 7, 'space group': 230}
    print('symmetry:', flag)
    sequence_length = 8501
    len_slice = {8501: 1}

    batch_size = 64
    num_epochs = 100
    lr = 0.001
    patience = 10
    patience_counter = 0
    best_test_acc = 0.0
    best_val_acc = 0.0

    loss_save_name = save_path + str(classes[flag]) + "_loss_curve.png"
    test_best_model_path = save_path + str(classes[flag]) + "_test_best_model" + ".pth"
    val_best_model_path = save_path + str(classes[flag]) + "_val_best_model" + ".pth"

    y_train, train_loader, val_loader, test_loader = dataset(batch_size, directory, ids_labels_path, sequence_length, len_slice, flag)

    model = MultiScaleCNN(classes_num=classes[flag], n_channel=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler()

    criterion = LabelSmoothingLoss(classes=classes[flag])

    loss_val = []
    loss_train = []
    for epoch in range(num_epochs):

        train_loss = train(model, train_loader, criterion, optimizer, device, scaler)
        loss_train.append(round(train_loss, 3))

        val_loss, val_class_accuracy, val_index = test(model, val_loader, criterion, device)
        loss_val.append(round(val_loss, 3))

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}")
        # val set result according to each epoch
        print('Val Accuracy, Val Macro Precision, Val Macro F1, Val PR-AUC(macro)')
        print(f"{val_index['accuracy']:.7f}, {val_index['macro_precision']:.7f}, {val_index['macro_f1']:.7f}, {val_index['PR_AUC_macro']:.7f}")

        # test set result according to each epoch
        test_loss, test_class_accuracy, test_index = test(model, test_loader, criterion, device)
        print('Test Accuracy, Test Macro Precision, Test Macro F1, Test PR-AUC(macro)')
        print(f"{test_index['accuracy']:.7f}, {test_index['macro_precision']:.7f}, {test_index['macro_f1']:.7f}, {test_index['PR_AUC_macro']:.7f}")

        ## for class accuracy
        # print(f"Epoch {epoch + 1}, Val Class Acc:{val_class_accuracy}")
        # print(f"Epoch {epoch + 1}, Test Class Acc:{test_class_accuracy}")

        scheduler.step(val_loss)
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # save val_best_model  or test_best_model
        # for val set
        if val_index['accuracy'] > best_val_acc:
            e_val = epoch + 1
            best_val_acc = val_index['accuracy']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss_train': loss_train,
                'loss_val': loss_val,
                'patience_counter': patience_counter,
                'best_test_acc': best_test_acc,
                'best_val_acc': best_val_acc
            }, val_best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
        # for test set
        if test_index['accuracy'] > best_test_acc:
            e_test = epoch + 1
            best_test_acc = test_index['accuracy']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss_train': loss_train,
                'loss_val': loss_val,
                'patience_counter': patience_counter,
                'best_test_acc': best_test_acc,
                'best_val_acc': best_val_acc
            }, test_best_model_path)

        if patience_counter >= patience:
            print("Early stopping triggered. Stopping training...")
            break

    print(f'val epoch:{e_val},best_val_acc:{round(best_val_acc, 7)},test epoch:{e_test},best_test_acc:{round(best_test_acc, 7)}')

    draw_loss(loss_train, loss_val, name1='Training Loss', name2='Validation Loss', save_name=loss_save_name)
    torch.cuda.empty_cache()


