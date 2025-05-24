#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluate.py
@Time    :   2023/08/03 19:35:43
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2023
@Desc    :   None
'''

# here put the import lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, RocCurveDisplay, \
                            accuracy_score, precision_score, recall_score, f1_score


def evaluate_intra_drift_robustness(model, x_test, y_test, metric_name=['f1', 'acc', 'fpr', 'fnr'], train_data=None, split_year=None):
    year_metrics = {}
    if train_data is not None:
        x_train, y_train = train_data
        year_metrics[split_year] = evaluate_model(model, x_train, y_train, metric_name=metric_name, legacy_format=False, check_binary=True)
    for year in x_test:
        data = x_test[year]
        label = y_test[year]
        metric_results = evaluate_model(model, data, label, metric_name=metric_name, legacy_format=False, check_binary=True)
        year_metrics[year] = metric_results
        print(f'[{year}] {metric_results}')
    results = pd.DataFrame(year_metrics)
    aut_results = results.apply(lambda row: aut_score(row.tolist()), axis=1)
    results['AUT'] = aut_results
    return results


def evaluate_concept_accuracy(model, X_test, y_concept, batch_size=32):
    yb_pred = model.get_predictions(X_test, 'exp', batch_size=batch_size)
    return compute_overall_concept_accuracy(yb_pred, y_concept)    


def evaluate_model(model, X_test, y_test, metric_name=['f1', 'acc'], y_concept=None, legacy_format=True, check_binary=False):
    if y_concept is None:
        y_pred = model.predict(X_test)
        concept_acc = None
    else:
        y_pred, yb_pred = model.get_predictions(X_test)
        concept_acc = compute_overall_concept_accuracy(yb_pred, y_concept)
    y_pred = np.argmax(y_pred, axis=1)
    if check_binary: y_test = multi2binary_malware_label(y_test)
    
    metric_result = {}
    if 'f1' in metric_name:
        metric_result['f1'] = f1_score(y_test, y_pred, average='weighted')
    if 'acc' in metric_name:
        accuracy = accuracy_score(y_test, y_pred)
        if legacy_format:
            metric_result['acc'] = (accuracy, concept_acc)
        else:
            metric_result['acc'] = accuracy
    if 'fpr' in metric_name or 'fnr' in metric_name:
        TN, FP, FN, TP = confusion_matrix_values(y_test, y_pred)
        if 'fpr' in metric_name:
            metric_result['fpr'] = FP / (FP + TN)
        if 'fnr' in metric_name:
            metric_result['fnr'] = FN / (FN + TP)
    
    if legacy_format:
        return metric_result.values()
    else:
        return metric_result


def evaluate_detection(anomaly_scores, y_test, drift_classes, top_k=50, figpath=None, verbose=False, drift_y_preds=None):
    y_true = multi2binary_drift_label(y_test, drift_classes)
    auc_score = draw_roc_curve(y_true, anomaly_scores, figpath=figpath, silent=True)
    y_true = np.array(y_true)
    anomaly_scores = np.array(anomaly_scores)
    top_k_indices = np.argpartition(-anomaly_scores, top_k)[:top_k]
    top_y_true = y_true[top_k_indices]
    detection_accuracy = accuracy_score(top_y_true, np.ones_like(top_y_true))
    if drift_y_preds is not None:
        detection_accuracy = (detection_accuracy, accuracy_score(y_true, drift_y_preds))
    if verbose:
        return auc_score, detection_accuracy, (y_true, anomaly_scores)
    return auc_score, detection_accuracy


def evaluate_uncertainty_detection(model, x_test, y_test, drift_classes, top_k=50, **kwargs):   
    if hasattr(model, 'get_drift_scores'):
        test_uncertainty = model.get_drift_scores(x_test, **kwargs)
    else:
        probs = model.predict(x_test)
        use_entropy = kwargs.get('use_entropy', True)
        # Multi-class classification; Higher scores indicate higher uncertainty.
        test_uncertainty = -np.sum(probs * np.log(probs + 1e-10), axis=1) if use_entropy\
             else 1 - (probs).max(-1)
    auc_score, accuracy, (y_true, y_pred) = evaluate_detection(test_uncertainty, y_test, drift_classes, top_k=top_k, verbose=True)
    return auc_score, accuracy, (y_true, y_pred)


def draw_roc_curve(y_true, y_scores, pos_label=1, figpath=None, silent=False):
    # print(y_true, y_scores, pos_label)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=pos_label)
    except ValueError:
        raise ValueError(f'NaN in y_true: {np.isnan(y_true).any()}, y_scores: {np.isnan(y_scores).any()}')
    roc_auc = auc(fpr, tpr)

    if not silent:
        plt.figure()
        lw = 2  # Line width
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  # Random classifier
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc="lower right")
        if figpath is None:
            plt.show()
        else:
            plt.savefig(figpath)
        plt.close()
    return roc_auc


def multi2binary_drift_label(y_true, newfamily):
    if type(newfamily) != list:
        newfamily = [newfamily]
    # y_true = y_true.apply(lambda x: 1 if x in newfamily else 0)
    y_true = [y in newfamily for y in y_true]
    return y_true


def multi2binary_malware_label(y):
    unique_values = np.unique(y)
    if len(unique_values) > 2 or not set(unique_values).issubset({0, 1}):
        return (y > 0).astype(int)
    return y


def draw_multiple_roc_curves(y_trues, y_preds, names, figpath=None, title_suffix='', **kwargs):
    color_lib = ['008000', 'db7093','004586','ffd320','5f5f5f','bf8040','83caff','314004'\
            ,'ccffcc', 'f5d6e0','cce6ff','fff5cc','e6e6e6','f2e6d9','cce9ff', 'f1fccf']
    num_curves = len(y_trues)
    draw_chance = [(i == num_curves - 1) if kwargs.pop('draw_chance', True) else False for i in range(num_curves)]
    draw_average= kwargs.pop('draw_average', False)
    if draw_average:
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        kwargs['lw'] = 1
        kwargs['alpha'] = 0.8
    
    _, ax = plt.subplots(figsize=(8, 6))
    for i in range(num_curves):
        viz = RocCurveDisplay.from_predictions(
            y_trues[i],
            y_preds[i],
            name=names[i],
            color=f'#{color_lib[i]}',
            ax=ax,
            plot_chance_level=draw_chance[i],
            **kwargs
        )
        if draw_average:
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

    if draw_average:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            # alpha=0.8,                                                                              
        )

    ax.set(
        xlim=[-0.02, 1.02],
        ylim=[-0.02, 1.02],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )
    # Set the font size for axis labels
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    # Set the font size for the tick labels
    ax.tick_params(axis='both', labelsize=13)
    # ax.axis("square")
    # Change the font color of the last curve in the legend
    ax.legend(loc="lower right", fontsize=14).get_texts()[-1].set_color('#b20000')     

    if title_suffix is not None:
        title = 'Mean ROC Curve' if draw_average else 'ROC Curves'
        plt.title(title + title_suffix, fontsize=16)

    if figpath is None:
        plt.show()
    else:
        plt.savefig(figpath)

    if draw_average: 
        return mean_auc, std_auc


def confusion_matrix_values(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TN, FP, FN, TP


def aut_score(metric_lst):
    """
    Compute the Area Under Time (AUT) metric.
    Parameters:
    - f: list of performance metrics (e.g., F1 scores) evaluated over time.
    Returns:
    - AUT value.
    """
    N = len(metric_lst)
    if N <= 1:
        raise ValueError("N should be greater than 1")  
    total_area = 0
    for k in range(N-1):
        total_area += (metric_lst[k+1] + metric_lst[k]) / 2
    return total_area / (N-1)


def compute_concept_metrics(yb_pred, yb_true):
    """
    calculate the four metrics for each concept
    """
    # Convert probabilities to binary labels using a threshold (e.g., 0.5)
    y_pred_binary = np.where(yb_pred > 0.5, 1, 0)
    # Compute metrics for each label
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    for i in range(yb_true.shape[1]):  # yb_true is a 2D array
        accuracies.append(accuracy_score(yb_true[:, i], y_pred_binary[:, i]))
        precisions.append(precision_score(yb_true[:, i], y_pred_binary[:, i], zero_division=1))
        recalls.append(recall_score(yb_true[:, i], y_pred_binary[:, i], zero_division=1))
        f1_scores.append(f1_score(yb_true[:, i], y_pred_binary[:, i], zero_division=1)) 
    return accuracies, precisions, recalls, f1_scores


def compute_overall_concept_accuracy(y_pred, y_test):
    # Convert probabilities to binary labels using a threshold (e.g., 0.5)
    y_pred_binary = np.where(y_pred > 0.5, 1, 0)
    # Flatten arrays
    y_test_flat = y_test.ravel()
    y_pred_binary_flat = y_pred_binary.ravel()
    # Compute overall binary accuracy
    accuracy = accuracy_score(y_test_flat, y_pred_binary_flat)  
    return accuracy


def count_correct_in_top_k(scores, labels, k):
    """
    Counts the number of correct drift samples within the top-k drifting scores.
    labels: 1 for drift, 0 for non-drift
    """
    combined_sorted = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    top_k = combined_sorted[:k]
    correct_count = sum(label for _, label in top_k)
    return correct_count
