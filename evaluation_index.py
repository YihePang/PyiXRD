from sklearn.metrics import precision_score, recall_score, f1_score, \
    matthews_corrcoef, precision_recall_curve, auc, confusion_matrix, cohen_kappa_score
import numpy as np

def class_acc(class_correct, class_total):
    class_accuracy = {}
    sorted_labels = sorted(class_correct.keys())
    for label in sorted_labels:
        class_accuracy[label] = round(class_correct[label] / class_total[label], 5)
    return class_accuracy

def accuracy(correct, total):
    return correct / total

def precision(index_flag, all_predicted, all_labels):
    return precision_score(all_labels, all_predicted, average=index_flag, zero_division=0)

def F1(index_flag, all_predicted, all_labels):
    return f1_score(all_labels, all_predicted, average=index_flag, zero_division=0)

def macro_pr_auc(all_probs, all_labels):
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    num_classes = all_probs.shape[1]
    pr_auc_macro = 0
    valid_classes = 0
    for i in range(num_classes):
        class_labels = (all_labels == i).astype(int)
        class_probs = all_probs[:, i]
        valid_probs_indices = ~np.isnan(class_probs)
        class_probs = class_probs[valid_probs_indices]
        class_labels = class_labels[valid_probs_indices]
        valid_labels_indices = ~np.isnan(class_labels)
        class_probs = class_probs[valid_labels_indices]
        class_labels = class_labels[valid_labels_indices]
        if len(class_probs) > 0:
            if np.sum(class_labels) > 0:
                precision, recall, _ = precision_recall_curve(class_labels, class_probs)
                pr_auc = auc(recall, precision)
                pr_auc_macro += pr_auc
                valid_classes += 1
    if valid_classes > 0:
        result = pr_auc_macro / valid_classes
    else:
        result = 0
    return result



