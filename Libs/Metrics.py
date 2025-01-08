"""
Module Name: Evaluation Metrics for Cascade Prediction Models
Developer: hwxu
Development Environment: Python 3.8+, PyTorch, NumPy, Scikit-learn
Date: July, 2024
Version: Bean
Description: This module provides a collection of metrics such as precision, recall, MAP, 
             MRR, and other ranking-based evaluations used to assess the performance 
             of cascade prediction models.
"""

import collections

import Constants
import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.preprocessing import label_binarize


def precision_at_k(relevance_score, k):
    """ 
    Computes precision at K given binary relevance scores.
    
    Args:
        relevance_score (list or array): Binary relevance scores.
        k (int): The rank position at which to compute precision.
    
    Returns:
        float: Precision at rank K.
    """
    assert k >= 1
    relevance_score = np.asarray(relevance_score)[:k] != 0
    if relevance_score.size != k:
        raise ValueError('Relevance score length < K')
    return np.mean(relevance_score)


def recall_at_k(relevance_score, k, m):
    """ 
    Computes recall at K given binary relevance scores.
    
    Args:
        relevance_score (list or array): Binary relevance scores.
        k (int): The rank position at which to compute recall.
        m (int): Total number of relevant entities.
    
    Returns:
        float: Recall at rank K.
    """
    assert k >= 1
    relevance_score = np.asarray(relevance_score)[:k] != 0
    if relevance_score.size != k:
        raise ValueError('Relevance score length < K')
    return np.sum(relevance_score) / float(m)


def mean_precision_at_k(relevance_scores, k):
    """ 
    Computes mean precision at K for multiple relevance scores.
    
    Args:
        relevance_scores (list of lists): List of binary relevance scores.
        k (int): The rank position at which to compute mean precision.
    
    Returns:
        float: Mean precision at rank K.
    """
    mean_p_at_k = np.mean([precision_at_k(r, k) for r in relevance_scores]).astype(np.float32)
    return mean_p_at_k


def mean_recall_at_k(relevance_scores, k, m_list):
    """ 
    Computes mean recall at K for multiple relevance scores.
    
    Args:
        relevance_scores (list of lists): List of binary relevance scores.
        k (int): The rank position at which to compute mean recall.
        m_list (list): List of relevant target entities for each data point.
    
    Returns:
        float: Mean recall at rank K.
    """
    mean_r_at_k = np.mean([recall_at_k(r, k, M) for r, M in zip(relevance_scores, m_list)]).astype(np.float32)
    return mean_r_at_k


def average_precision(relevance_score, K, m):
    """ 
    Computes average precision for relevance scores.
    
    Args:
        relevance_score (list or array): Binary relevance scores.
        K (int): Maximum rank position.
        m (int): Total number of relevant entities.
    
    Returns:
        float: Average precision.
    """
    r = np.asarray(relevance_score) != 0
    out = [precision_at_k(r, k + 1) for k in range(0, K) if r[k]]
    if not out:
        return 0.
    return np.sum(out) / float(min(K, m))


def MAP(relevance_scores, k, m_list):
    """ 
    Computes Mean Average Precision (MAP) for multiple relevance scores.
    
    Args:
        relevance_scores (list of lists): List of binary relevance scores.
        k (int): Maximum rank position.
        m_list (list): List of relevant target entities for each data point.
    
    Returns:
        float: Mean Average Precision.
    """
    map_val = np.mean([average_precision(r, k, M) for r, M in zip(relevance_scores, m_list)]).astype(np.float32)
    return map_val


def MRR(relevance_scores):
    """ 
    Computes Mean Reciprocal Rank (MRR).
    
    Args:
        relevance_scores (list of lists): List of binary relevance scores.
    
    Returns:
        float: Mean reciprocal rank.
    """
    rs = (np.asarray(r).nonzero()[0] for r in relevance_scores)
    mrr_val = np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs]).astype(np.float32)
    return mrr_val


def get_masks(top_k, inputs):
    """ 
    Creates a mask to filter out dummy sequences where seed size is 1.
    
    Args:
        top_k (array): Array of top-K predictions.
        inputs (list of lists): List of seed users.
    
    Returns:
        np.array: Binary mask array.
    """
    masks = [(0 if len(set(inputs[i])) == 1 and list(set(inputs[i]))[0] == 0 else 1) for i in range(top_k.shape[0])]
    return np.array(masks).astype(np.int32)


def remove_seeds(top_k, inputs):
    """ 
    Replaces seed users from top-K predictions with -1.
    
    Args:
        top_k (array): Array of top-K predictions.
        inputs (list of lists): List of seed users.
    
    Returns:
        np.array: Updated top-K predictions with seeds replaced by -1.
    """
    result = []
    for i in range(top_k.shape[0]):
        seeds = set(inputs[i])
        lst = list(top_k[i])  # Top-K predicted users
        lst = [u for u in lst if u not in seeds]
        lst += [-1] * (len(top_k[i]) - len(lst))  # Replace seeds with -1
        result.append(lst)
    return np.array(result).astype(np.int32)


def get_relevance_scores(top_k_filter, targets):
    """ 
    Creates binary relevance scores by checking if top-K predicted users are in the target set.
    
    Args:
        top_k_filter (array): Filtered top-K predictions.
        targets (list of lists): List of target users.
    
    Returns:
        np.array: Binary relevance scores.
    """
    return np.array([np.isin(top_k_filter[i], targets[i]) for i in range(top_k_filter.shape[0])])


def one_hot(values, num_classes):
    """ 
    Converts a batch of sequences into one-hot encoding.
    
    Args:
        values (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
        num_classes (int): Total number of classes for one-hot encoding.
    
    Returns:
        torch.Tensor: One-hot encoded tensor of shape (batch_size, num_classes).
    """
    batch_size = values.shape[0]
    result = torch.zeros(batch_size, num_classes)
    for i in range(batch_size):
        for j in range(values.shape[1]):
            if values[i][j] != -1:
                result[i][values[i][j]] = 1
    return result.long()


def masked_select(inputs, masks):
    """ 
    Selects elements from inputs based on binary masks.
    
    Args:
        inputs (list): List of input elements.
        masks (list): List of binary masks (1 or 0).
    
    Returns:
        np.array: Masked selected inputs.
    """
    return np.array([inputs[i] for i, mask in enumerate(masks) if mask == 1]).astype(np.int32)


def _retype(y_prob, y):
    """ 
    Ensures that y_prob and y are both sequences or numpy arrays.
    
    Args:
        y_prob (array or list): Prediction probabilities.
        y (array or list): True labels.
    
    Returns:
        tuple: (y_prob, y) as numpy arrays.
    """
    if not isinstance(y, (collections.Sequence, np.ndarray)):
        y_prob = [y_prob]
        y = [y]
    return np.array(y_prob), np.array(y)


def _binarize(y, n_classes=None):
    """ 
    Converts labels into binary format using one-hot encoding.
    
    Args:
        y (list or array): Labels.
        n_classes (int): Number of classes for binarization.
    
    Returns:
        np.array: Binary labels.
    """
    return label_binarize(y, classes=range(n_classes))


def apk(actual, predicted, k=10):
    """
    Computes the average precision at K.
    
    Args:
        actual (list): Ground truth elements.
        predicted (list): Predicted elements in ranked order.
        k (int): Maximum number of predicted elements.
    
    Returns:
        float: Average precision at K.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k) if actual else 0.0


def mapk(y_prob, y, k=10):
    """
    Computes mean average precision at K.
    
    Args:
        y_prob (list of lists): Prediction probabilities.
        y (list): Ground truth labels.
        k (int): Maximum number of predicted elements.
    
    Returns:
        float: Mean average precision at K.
    """
    predicted = [np.argsort(p_)[-k:][::-1] for p_ in y_prob]
    actual = [[y_] for y_ in y]
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def mean_rank(y_prob, y):
    """ 
    Computes the mean rank of the true labels in the predicted probabilities.
    
    Args:
        y_prob (array): Prediction probabilities.
        y (array): True labels.
    
    Returns:
        float: Mean rank of true labels.
    """
    ranks = [y_prob.shape[1] - rankdata(p_, method='max')[y_] for p_, y_ in zip(y_prob, y)]
    return sum(ranks) / len(ranks)


def hits_k(y_prob, y, k=10):
    """ 
    Computes the hits@k metric, which checks if the true label is in the top-k predictions.
    
    Args:
        y_prob (array): Prediction probabilities.
        y (array): True labels.
        k (int): Number of top predictions to check.
    
    Returns:
        float: Proportion of true labels in the top-k predictions.
    """
    return np.mean([1. if y_ in p_.argsort()[-k:][::-1] else 0. for p_, y_ in zip(y_prob, y)])


def portfolio(pred, gold, k_list=[1, 5, 10, 20]):
    """
    Computes multiple ranking metrics for a portfolio of predictions.
    
    Args:
        pred (array): Predicted probabilities.
        gold (array): Ground truth labels.
        k_list (list): List of K values to compute metrics at.
    
    Returns:
        tuple: Dictionary of metric scores and total score length.
    """
    scores_len = 0
    y_prob = []
    y = []
    for i in range(gold.shape[0]):
        if gold[i] != Constants.PAD:
            scores_len += 1.0
            y_prob.append(pred[i])
            y.append(gold[i])
    
    scores = {f'hits@{k}': hits_k(y_prob, y, k=k) for k in k_list}
    scores.update({f'map@{k}': mapk(y_prob, y, k=k) for k in k_list})

    return scores, scores_len


def compute_metric(y_prob, y_true, k_list=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
    """
    Computes various ranking metrics for the provided predictions and ground truth.
    
    Args:
        y_prob (array): Predicted probabilities.
        y_true (array): Ground truth labels.
        k_list (list): List of K values to compute metrics at.
    
    Returns:
        tuple: Dictionary of metric scores and total score length.
    """
    scores_len = 0
    y_prob = np.array(y_prob)
    y_true = np.array(y_true)

    scores = {f'hits@{k}': [] for k in k_list}
    scores.update({f'map@{k}': [] for k in k_list})

    for p_, y_ in zip(y_prob, y_true):
        if y_ != 0:  # Assuming '0' is a placeholder for PAD
            scores_len += 1.0
            p_sort = p_.argsort()
            for k in k_list:
                topk = p_sort[-k:][::-1]
                scores[f'hits@{k}'].append(1. if y_ in topk else 0.)
                scores[f'map@{k}'].append(apk([y_], topk, k))

    scores = {k: np.mean(v) for k, v in scores.items()}
    return scores, scores_len
