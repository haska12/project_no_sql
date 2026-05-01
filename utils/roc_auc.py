import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from itertools import cycle

def compute_topk_accuracy(y_true, y_probs, k=2):
    """Top‑k accuracy: true class in top k predicted probabilities"""
    topk_preds = np.argsort(y_probs, axis=1)[:, -k:]
    correct = sum([y_true[i] in topk_preds[i] for i in range(len(y_true))])
    return correct / len(y_true)

def plot_multiclass_roc(y_true, y_probs, class_names, save_path=None):
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(8,6))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0,1], [0,1], 'k--', lw=1)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
    return plt.gcf(), roc_auc