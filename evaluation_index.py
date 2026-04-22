from sklearn.metrics import precision_score, recall_score, f1_score,matthews_corrcoef,precision_recall_curve, auc,confusion_matrix, cohen_kappa_score
import numpy as np

def class_acc(class_correct,class_total):
    class_accuracy = {}
    # print('class correct:',class_correct)
    # print('class total:',class_total)
    sorted_labels = sorted(class_correct.keys())
    for label in sorted_labels:
        class_accuracy[label] = round(class_correct[label] / class_total[label],5)
    return class_accuracy

def accuracy(correct,total):
    return correct / total

def precision(index_flag,all_predicted,all_labels):
    return precision_score(all_labels, all_predicted, average=index_flag, zero_division=0)

# def recall(index_flag,all_predicted,all_labels):
#     return recall_score(all_labels, all_predicted, average=index_flag, zero_division=0)
def recall(index_flag, all_predicted, all_labels):
    # 检查 all_labels 中是否有正类样本
    if sum(all_labels) == 0:
        return 0
    return recall_score(all_labels, all_predicted, average=index_flag, zero_division=0)

def F1(index_flag,all_predicted,all_labels):
    return f1_score(all_labels, all_predicted, average=index_flag, zero_division=0)

def MCC_index(all_predicted,all_labels):
    return matthews_corrcoef(all_labels, all_predicted)

def macro_pr_auc(all_probs, all_labels):
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    # 处理多分类的 PR - AUC（宏观平均）
    num_classes = all_probs.shape[1]
    pr_auc_macro = 0
    valid_classes = 0  # 记录有效类别的数量
    for i in range(num_classes):
        # 提取当前类别的真实标签和预测概率
        class_labels = (all_labels == i).astype(int)
        class_probs = all_probs[:, i]
        # 检查并处理 class_probs 中的 NaN 值
        valid_probs_indices = ~np.isnan(class_probs)
        class_probs = class_probs[valid_probs_indices]
        class_labels = class_labels[valid_probs_indices]
        # 检查并处理 class_labels 中的 NaN 值
        valid_labels_indices = ~np.isnan(class_labels)
        class_probs = class_probs[valid_labels_indices]
        class_labels = class_labels[valid_labels_indices]
        if len(class_probs) > 0:
            if np.sum(class_labels) > 0:
                # 计算精确率 - 召回率曲线
                precision, recall, _ = precision_recall_curve(class_labels, class_probs)
                # 计算当前类别的 PR - AUC
                pr_auc = auc(recall, precision)
                pr_auc_macro += pr_auc
                valid_classes += 1
    # 计算宏观平均的 PR - AUC
    if valid_classes > 0:
        result = pr_auc_macro / valid_classes
    else:
        result = 0
    return result

def G_mean(all_predicted,all_labels):
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predicted)
    num_classes = cm.shape[0]
    # 计算每个类别的召回率（真阳性率）
    recalls = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recalls.append(recall)
    # 计算 G - mean
    gmean = np.sqrt(np.prod(recalls))
    return gmean

def cohen_kappa(all_predicted,all_labels):
    return cohen_kappa_score(all_labels, all_predicted)

'''
#会出NAN的错误
# def macro_pr_auc(all_probs,all_labels):
#     all_labels = np.array(all_labels)
#     all_probs = np.array(all_probs)
#     # 处理多分类的 PR - AUC（宏观平均）
#     num_classes = all_probs.shape[1]
#     pr_auc_macro = 0
#     for i in range(num_classes):
#         # 提取当前类别的真实标签和预测概率
#         class_labels = (all_labels == i).astype(int)
#         class_probs = all_probs[:, i]
#         # 计算精确率 - 召回率曲线
#         precision, recall, _ = precision_recall_curve(class_labels, class_probs)
#         # 计算当前类别的 PR - AUC
#         pr_auc = auc(recall, precision)
#         pr_auc_macro += pr_auc
#     # 计算宏观平均的 PR - AUC
#     result=pr_auc_macro / num_classes
#     return result

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

    all_predicted0 = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs,features = model(inputs)
            # loss = criterion(outputs, features, labels)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            all_predicted0.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
            # for acc
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # for precision recall f1 MCC gmean kappa cm
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
      
    # 计算混淆矩阵
    num_classes = 7
    conf_matrix = confusion_matrix(all_labels, all_predicted0)
    print("Confusion Matrix:")
    print(conf_matrix)

    loss = running_loss / len(test_loader)
    class_accuracy = class_acc(class_correct, class_total)
    acc = accuracy(correct, total)
    weighted_pre = precision(average_flag[2], all_predicted, all_labels0)
    macro_pre = precision(average_flag[1], all_predicted, all_labels0)
    weighted_rec = recall(average_flag[2], all_labels0, all_predicted)
    macro_rec = recall(average_flag[1], all_labels0, all_predicted)
    weighted_f1 = F1(average_flag[2], all_labels0, all_predicted)
    macro_f1 = F1(average_flag[1], all_labels0, all_predicted)
    MCC = MCC_index(all_predicted, all_labels0)
    pr_auc_macro = macro_pr_auc(all_probs, all_labels1)
    kappa = cohen_kappa(all_predicted, all_labels0)

    index = {}
    name = ['accuracy', 'weighted_precision', 'macro_precision', 'weighted_recall', 'macro_recall', 'weighted_f1',
            'macro_f1', 'MCC', 'PR_AUC_macro', 'kappa']
    all_index = [acc, weighted_pre, macro_pre, weighted_rec, macro_rec, weighted_f1, macro_f1, MCC, pr_auc_macro, kappa]
    for i in range(len(name)):
        index[name[i]] = all_index[i]

    return loss, class_accuracy, index
'''
