"""
Module Name: Cascade Prediction Training Script
Developer: hwxu
Development Environment: Python 3.8+, PyTorch, NumPy
Date: August, 2024
Version: Bean
Description: End-to-end pipeline for training, validating, and testing cascade prediction models.
             This script loads data, initializes the model, and runs the training process.
"""

import argparse
import operator
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from DataSet import *
from HypergraphUtil import *
from Metrics import *
from Module import *

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dataset_name', default='christianity')
parser.add_argument('-epoch', default=50)
parser.add_argument('-batch_size', default=64)
parser.add_argument('-emb_dim', default=64)
parser.add_argument('-train_rate', default=0.8)
parser.add_argument('-valid_rate', default=0.1)
# Hyperparameter for balancing micro/macro tasks
parser.add_argument('-lambda_loss', default=0.3)
# Hyperparameter for orthogonality constraint
parser.add_argument('-gamma_loss', default=0.05)
parser.add_argument('-max_seq_length', default=200)
parser.add_argument('-step_split', default=8)  # Number of cascade hypergraphs
parser.add_argument('-lr', default=0.001)  # Learning rate
parser.add_argument('-early_stop_step', default=10)  # Early stopping criterion
opt = parser.parse_args()


def MAE(y, y_predicted):
    """
    Computes Mean Absolute Error (MAE) between predicted and true values.

    Args:
        y (torch.Tensor): Ground truth labels.
        y_predicted (torch.Tensor): Model predicted values.

    Returns:
        torch.Tensor: MAE value.
    """
    y_predicted = y_predicted.squeeze()
    mae = torch.mean(torch.abs(y_predicted - y))
    return mae


def MSLE(y, y_predicted):
    """
    Computes Mean Squared Logarithmic Error (MSLE) between predicted and true values.

    Args:
        y (torch.Tensor): Ground truth labels.
        y_predicted (torch.Tensor): Model predicted values.

    Returns:
        float: MSLE value.
    """
    predicted = y_predicted.cpu().detach().numpy().squeeze()
    predicted[predicted < 1] = 1
    label = y.cpu().detach().numpy()
    msle = np.square(np.log2(predicted) - np.log2(label))
    return np.mean(msle)


def get_previous_user_mask(seq, user_size):
    """
    Creates a mask for previous activated users to prevent them from being activated again.

    Args:
        seq (torch.Tensor): Input sequences (batch_size, sequence_length).
        user_size (int): Total number of users.

    Returns:
        torch.Tensor: Masked sequences with previously activated users masked.
    """
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(
        seq.size(0), seq.size(1), seq.size(1))
    previous_mask = torch.from_numpy(
        np.tril(np.ones(prev_shape)).astype('float32'))

    if seq.is_cuda:
        previous_mask = previous_mask.cuda()

    masked_seq = previous_mask * seqs.float()
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)

    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float('-inf'))
    return masked_seq


def get_performance(crit, pred, gold):
    """
    Computes the loss and number of correct predictions.

    Args:
        crit (nn.Module): Loss function (e.g., CrossEntropyLoss).
        pred (torch.Tensor): Predicted user probabilities (batch_size * seq_len, user_size).
        gold (torch.Tensor): True user indices (batch_size, seq_len).

    Returns:
        tuple: (loss, number of correct predictions)
    """
    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data).masked_select(
        gold.ne(Constants.PAD).data).sum().float()
    return loss, n_correct


def train_epoch(model, train_loader, relation_graph, hypergraph_list, micro_loss_func, optimizer, lambda_loss, gamma_loss, user_size, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The cascade prediction model.
        train_loader (DataLoader): Training data loader.
        relation_graph: The social graph.
        hypergraph_list: List of dynamic hypergraphs.
        micro_loss_func (nn.Module): Loss function for micro predictions.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        lambda_loss (float): Weight for macro task loss.
        gamma_loss (float): Weight for orthogonality constraint loss.
        user_size (int): Total number of users.
        device (torch.device): Device for computation.

    Returns:
        tuple: (average loss, accuracy)
    """
    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0

    for i, batch in enumerate(train_loader):
        tgt, tgt_timestamp, tgt_idx, tgt_len = (
            item.to(device) for item in batch)
        gold = tgt[:, 1:]  # Target users for each time step

        n_words = gold.data.ne(Constants.PAD).sum().float()
        n_total_words += n_words

        pred_micro, pred_macro, loss_adv, loss_diff = model(
            hypergraph_list, relation_graph, tgt)
        mask = get_previous_user_mask(tgt[:, :-1].cpu(), user_size).to(device)
        micro_loss, n_correct = get_performance(
            micro_loss_func, (pred_micro[:, :-1, :] + mask).view(-1, pred_micro.size(-1)), gold)
        macro_loss = MAE(tgt_len, pred_macro)

        loss = (1 - lambda_loss) * micro_loss + lambda_loss * \
            macro_loss + loss_adv + gamma_loss * loss_diff

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_total_correct += n_correct
        total_loss += loss.item()

    return total_loss / n_total_words, n_total_correct / n_total_words


def test_epoch(model, data_loader, relation_graph, hypergraph_list, user_size, device, k_list=[10, 50, 100]):
    """
    Evaluates the model on a test dataset.

    Args:
        model (nn.Module): The trained cascade prediction model.
        data_loader (DataLoader): Data loader for evaluation.
        relation_graph: The social graph.
        hypergraph_list: List of dynamic hypergraphs.
        user_size (int): Total number of users.
        device (torch.device): Device for computation.
        k_list (list): List of K values for evaluation metrics.

    Returns:
        tuple: (micro scores, macro scores)
    """
    model.eval()

    macro_metric = {}
    scores = {f'hits@{k}': 0 for k in k_list}
    scores.update({f'map@{k}': 0 for k in k_list})
    msle = []

    n_total_words = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            tgt, tgt_timestamp, tgt_idx, tgt_len = (
                item.to(device) for item in batch)
            y_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()

            pred_micro, pred_macro, loss_adv, loss_diff = model(
                hypergraph_list, relation_graph, tgt)
            mask = get_previous_user_mask(
                tgt[:, :-1].cpu(), user_size).to(device)
            y_pred = (pred_micro[:, :-1, :] + mask).view(-1,
                                                         pred_micro.size(-1)).detach().cpu().numpy()

            scores_batch, scores_len = compute_metric(y_pred, y_gold, k_list)
            n_total_words += scores_len

            for k in k_list:
                scores[f'hits@{k}'] += scores_batch[f'hits@{k}'] * scores_len
                scores[f'map@{k}'] += scores_batch[f'map@{k}'] * scores_len

            msle.append(MSLE(tgt_len, pred_macro))

    for k in k_list:
        scores[f'hits@{k}'] /= n_total_words
        scores[f'map@{k}'] /= n_total_words

    macro_metric['MSLE'] = np.mean(msle)

    return scores, macro_metric


def main():
    """
    Main function for training, validation, and testing of the cascade prediction model.
    Loads data, prepares the model, and runs the training loop.
    """
    # Load dataset and parameters
    dataset = opt.dataset_name
    max_seq_length = opt.max_seq_length
    batch_size = opt.batch_size
    emb_dim = opt.emb_dim
    step_split = opt.step_split
    lambda_loss = opt.lambda_loss
    gamma_loss = opt.gamma_loss
    early_stop_step = opt.early_stop_step
    lr = opt.lr
    epoch = opt.epoch
    train_rate = opt.train_rate
    valid_rate = opt.valid_rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    user_size, total_cascades, timestamps, train, valid, test = split_data(
        dataset, train_rate, valid_rate, load_dict=True)
    train_loader = DataLoader(train, batch_size, load_dict=True, cuda=False)
    valid_loader = DataLoader(valid, batch_size, load_dict=True, cuda=False)
    test_loader = DataLoader(test, batch_size, load_dict=True, cuda=False)

    # Prepare model
    relation_graph = build_relation_graph(dataset, device)
    hypergraph_list = build_dynamic_cascade_hypergraph(
        total_cascades, timestamps, user_size, device, step_split)
    model = Module(user_size, emb_dim, step_split,
                   max_seq_length, 2, device).to(device)

    # Loss function and optimizer
    micro_loss_func = nn.CrossEntropyLoss(
        size_average=False, ignore_index=Constants.PAD)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Track the best scores
    k_list = [10, 50, 100]
    micro_score = float('-inf')
    macro_score = float('inf')
    micro_best_epoch = 0
    macro_best_epoch = 0

    # Print initial parameter details
    print(f"{'=' * 60}")
    print(f"{'Parameter Details':^60}")
    print(f"{'-' * 60}")
    for arg, value in vars(opt).items():
        print(f"{arg:>30}: {value}")
    print(f"{'=' * 60}\n")

    total_time = 0
    for epoch_i in range(epoch):
        print(f"{'=' * 60}")
        print(f"{'Epoch':^60}")
        print(f"Epoch {epoch_i + 1}/{epoch}")
        print(f"{'-' * 60}")

        # Training
        start = time.time()
        loss, train_micro_accu = train_epoch(
            model, train_loader, relation_graph, hypergraph_list, micro_loss_func, optimizer, lambda_loss, gamma_loss, user_size, device)
        end = time.time()
        total_time += end - start

        print(f"{'Training':^60}")
        print(f"Mean Prediction Loss: {loss:.4f}")
        print(f"Training Time: {end - start:.2f} seconds\n")

        # Validation
        scores, macro_metric = test_epoch(
            model, valid_loader, relation_graph, hypergraph_list, user_size, device, k_list)
        print(f"{'Validation':^60}")
        print(f"{'Micro Prediction Results':^60}")
        for k in k_list:
            print(f"Hits@{k}: {scores[f'hits@{k}']:.4f} | MAP@{k}: {scores[f'map@{k}']:.4f}")
        print(f"{'Macro Prediction Results':^60}")
        print(f"MSLE: {macro_metric['MSLE']:.4f}\n")

        # Test
        scores, macro_metric = test_epoch(
            model, test_loader, relation_graph, hypergraph_list, user_size, device, k_list)
        print(f"{'Test':^60}")
        print(f"{'Micro Prediction Results':^60}")
        for k in k_list:
            print(f"Hits@{k}: {scores[f'hits@{k}']:.4f} | MAP@{k}: {scores[f'map@{k}']:.4f}")
        print(f"{'Macro Prediction Results':^60}")
        print(f"MSLE: {macro_metric['MSLE']:.4f}\n")

        if scores['map@100'] > micro_score:
            micro_score = scores['map@100']
            micro_best_epoch = epoch_i + 1

        if macro_metric['MSLE'] < macro_score:
            macro_score = macro_metric['MSLE']
            macro_best_epoch = epoch_i + 1

    print(f"{'=' * 60}")
    print(f"{'Best Results':^60}")
    print(f"{'-' * 60}")
    print(f"Best Micro Prediction Epoch: {micro_best_epoch}")
    print(f"Best Micro MAP@100 Score: {micro_score:.4f}")
    print(f"Best Macro Prediction Epoch: {macro_best_epoch}")
    print(f"Best Macro MSLE Score: {macro_score:.4f}")
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
