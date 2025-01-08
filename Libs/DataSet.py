"""
Module Name: DataSet
Developer: hwxu
Development Environment: Python 3.8+, PyTorch, NumPy
Date: June, 2024
Version: Bean
Description: Data loading, preprocessing, and training pipeline for cascade prediction models.
"""

import random
import numpy as np
import torch
from torch.autograd import Variable
import Constants
import pickle

class Options:
    """
    Options class to hold various configuration details like file paths and dimensions.
    """
    def __init__(self, data_name='douban'):
        self.data = f'data/{data_name}/cascades.txt'
        self.u2idx_dict = f'data/{data_name}/u2idx.pickle'
        self.idx2u_dict = f'data/{data_name}/idx2u.pickle'
        self.save_path = ''
        self.net_data = f'data/{data_name}/edges.txt'
        self.embed_dim = 64


def split_data(data_name, train_rate=0.8, valid_rate=0.1, random_seed=300, load_dict=True, with_EOS=True):
    """
    Split the data into train, validation, and test sets based on provided parameters.
    
    Args:
        data_name (str): Name of the dataset.
        train_rate (float): Proportion of data used for training.
        valid_rate (float): Proportion of data used for validation.
        random_seed (int): Seed for shuffling the data.
        load_dict (bool): Whether to load pre-saved dictionaries.
        with_EOS (bool): Whether to include the end-of-sequence token.

    Returns:
        tuple: User size, cascades, timestamps, train, validation, and test sets.
    """
    options = Options(data_name)

    # Load or build user-to-index and index-to-user mappings
    if load_dict:
        with open(options.u2idx_dict, 'rb') as f:
            u2idx = pickle.load(f)
        with open(options.idx2u_dict, 'rb') as f:
            idx2u = pickle.load(f)
    else:
        user_size, u2idx, idx2u = build_index(options.data)
        with open(options.u2idx_dict, 'wb') as f:
            pickle.dump(u2idx, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(options.idx2u_dict, 'wb') as f:
            pickle.dump(idx2u, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Initialize cascade data
    cascades, timestamps = [], []
    for line in open(options.data):
        if not line.strip():
            continue
        userlist, timestamplist = parse_cascade_line(line, u2idx, data_name)
        if 1 < len(userlist) <= 500:  # Filter cascade length
            if with_EOS:
                userlist.append(Constants.EOS)
                timestamplist.append(Constants.EOS)
            cascades.append(userlist)
            timestamps.append(timestamplist)

    # Sort cascades by start time
    order = sorted(range(len(timestamps)), key=lambda x: timestamps[x])
    cascades = [cascades[i] for i in order]
    timestamps = [timestamps[i] for i in order]

    # Split data into train, validation, and test sets
    train_idx = int(train_rate * len(cascades))
    valid_idx = int((train_rate + valid_rate) * len(cascades))

    train_data = (cascades[:train_idx], timestamps[:train_idx], list(range(train_idx)))
    valid_data = (cascades[train_idx:valid_idx], timestamps[train_idx:valid_idx], list(range(train_idx, valid_idx)))
    test_data = (cascades[valid_idx:], timestamps[valid_idx:], list(range(valid_idx, len(cascades))))

    # Shuffle the training set
    random.seed(random_seed)
    for dataset in train_data:
        random.shuffle(dataset)

    user_size = len(u2idx)
    print_summary(cascades, train_data, valid_data, test_data, user_size)

    return user_size, cascades, timestamps, train_data, valid_data, test_data


def parse_cascade_line(line, u2idx, data_name):
    """
    Parse a line from the cascade file to extract user and timestamp information.

    Args:
        line (str): Line from the data file.
        u2idx (dict): Dictionary mapping users to indices.
        data_name (str): Name of the dataset for format adjustments.

    Returns:
        tuple: List of users and list of timestamps.
    """
    userlist, timestamplist = [], []
    chunks = line.strip().split(',' if data_name != 'memetracker' else ' ')
    for chunk in chunks:
        try:
            user, timestamp = chunk.split()[:2]
            if user in u2idx:
                userlist.append(u2idx[user])
                timestamplist.append(float(timestamp))
        except ValueError:
            continue
    return userlist, timestamplist


def build_index(data_path):
    """
    Build user-to-index and index-to-user mappings from the data.

    Args:
        data_path (str): Path to the data file.

    Returns:
        tuple: Number of users, u2idx (user-to-index), idx2u (index-to-user).
    """
    user_set = set()
    u2idx = {'<blank>': 0, '</s>': 1}
    idx2u = ['<blank>', '</s>']

    for line in open(data_path):
        for chunk in line.strip().split(','):
            user = chunk.split()[0]
            user_set.add(user)

    for pos, user in enumerate(user_set, start=2):
        u2idx[user] = pos
        idx2u.append(user)

    user_size = len(u2idx)
    print(f"User size: {user_size}")
    return user_size, u2idx, idx2u


def print_summary(cascades, train, valid, test, user_size):
    """
    Print the summary statistics of the data.

    Args:
        cascades (list): List of all cascades.
        train (tuple): Training data.
        valid (tuple): Validation data.
        test (tuple): Test data.
        user_size (int): Size of the user set.
    """
    total_len = sum(len(cas) - 1 for cas in cascades)
    print(f"Training size: {len(train[0])}\nValidation size: {len(valid[0])}\nTesting size: {len(test[0])}")
    print(f"Total cascades: {len(cascades)}")
    print(f"Average length: {total_len / len(cascades):.2f}")
    print(f"Max length: {max(len(cas) for cas in cascades)}")
    print(f"Min length: {min(len(cas) for cas in cascades)}")
    print(f"User size: {user_size - 2}")  # Exclude special tokens


class DataLoader:
    """
    DataLoader class to iterate over cascades in batches for training/testing.

    Args:
        cas (tuple): Tuple containing cascades, timestamps, and indices.
        batch_size (int): Number of cascades per batch.
        load_dict (bool): Unused parameter, included for future flexibility.
        cuda (bool): Whether to use GPU for tensor operations.
        test (bool): Whether to operate in test mode.
        with_EOS (bool): Whether to include EOS token.
    """
    def __init__(self, cas, batch_size=64, load_dict=True, cuda=True, test=False, with_EOS=True):
        self.batch_size = batch_size
        self.cascades, self.timestamps, self.indices = cas
        self.lengths = [len(cas) for cas in self.cascades]
        self.test = test
        self.with_EOS = with_EOS
        self.cuda = cuda
        self.num_batches = int(np.ceil(len(self.cascades) / batch_size))
        self.iter_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()

    def __len__(self):
        return self.num_batches

    def next_batch(self):
        """
        Get the next batch of cascades, timestamps, and indices, padding as necessary.

        Returns:
            tuple: Padded batch of cascades, timestamps, indices, and lengths.
        """
        if self.iter_count >= self.num_batches:
            self.iter_count = 0
            raise StopIteration()

        start_idx = self.iter_count * self.batch_size
        end_idx = (self.iter_count + 1) * self.batch_size
        self.iter_count += 1

        cascades_batch = self.pad_to_longest(self.cascades[start_idx:end_idx])
        timestamps_batch = self.pad_to_longest(self.timestamps[start_idx:end_idx])
        indices_batch = Variable(torch.LongTensor(self.indices[start_idx:end_idx]), volatile=self.test)
        lengths_batch = Variable(torch.LongTensor(self.lengths[start_idx:end_idx]), volatile=self.test)

        return cascades_batch, timestamps_batch, indices_batch, lengths_batch

    def pad_to_longest(self, insts):
        """
        Pad sequences in the batch to the longest sequence length (max 200).

        Args:
            insts (list): List of sequences.

        Returns:
            torch.Tensor: Padded tensor of sequences.
        """
        max_len = 200
        padded_insts = np.array([
            inst + [Constants.PAD] * (max_len - len(inst)) if len(inst) < max_len else inst[:max_len]
            for inst in insts
        ])

        inst_tensor = Variable(torch.LongTensor(padded_insts), volatile=self.test)
        return inst_tensor.cuda() if self.cuda else inst_tensor
