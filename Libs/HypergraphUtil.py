"""
Module Name: Graph Construction for Cascade Models
Developer: hwxu
Development Environment: Python 3.8+, PyTorch, DHG, NumPy
Date: July, 2024
Version: Bean
Description: This module handles the construction of relation graphs and hypergraphs from cascade data, enabling dynamic cascade hypergraph creation for modeling social networks and user interactions over time.
"""

import os
import pickle

import dhg
import torch
from DataSet import *

# from DataLoader import *


def build_relation_graph(data_name, device):
    """
    Constructs a social relation graph based on the input data.
    
    Args:
        data_name (str): Name of the dataset.
        device (str): The device ('cpu' or 'cuda') where the graph will be stored.
    
    Returns:
        dhg.Graph: A directed graph object representing user relations.
    """
    data = Options(data_name)

    # Load user-to-index mapping
    with open(data.u2idx_dict, 'rb') as f:
        u2idx = pickle.load(f)

    # Load edges if network data exists
    if os.path.exists(data.net_data):
        with open(data.net_data, 'r') as f:
            edge_list = f.read().strip().split('\n')
            if data_name in ['douban', 'twitter']:
                edge_list = [edge.split(',') for edge in edge_list]
            else:
                edge_list = [edge.split(' ') for edge in edge_list]

            # Map users to indices, filtering out invalid edges
            edge_list = [(u2idx[edge[0]], u2idx[edge[1]]) for edge in edge_list
                         if edge[0] in u2idx and edge[1] in u2idx]
    else:
        return None

    # Construct graph and log information
    user_size = len(u2idx)
    relation_graph = dhg.Graph(user_size, edge_list, device=device)
    print(f'#Link: {len(relation_graph.e[0])}')
    return relation_graph


def build_cascade_hypergraph(cascades, user_size, device):
    """
    Constructs a hypergraph from cascades where each cascade is an edge.
    
    Args:
        cascades (list): List of user cascades.
        user_size (int): Number of users in the dataset.
        device (str): The device ('cpu' or 'cuda') where the hypergraph will be stored.
    
    Returns:
        dhg.Hypergraph: A hypergraph object representing the cascades.
    """
    edge_list = []
    for cascade in cascades:
        cascade = set(cascade)
        cascade.discard(0)  # Remove zero from cascade if it exists
        edge_list.append(cascade)

    return dhg.Hypergraph(user_size, edge_list, device=device)


def build_dynamic_cascade_hypergraph(examples, examples_times, user_size, device, step_split=8):
    """
    Dynamically builds a sequence of hypergraphs by splitting time intervals for cascades.
    
    Args:
        examples (list): Cascades (users).
        examples_times (list): Timestamps of user participation in the cascades.
        user_size (int): Total number of users in the dataset.
        device (str): The device ('cpu' or 'cuda') where the hypergraph will be stored.
        step_split (int): Number of hypergraphs to divide the time intervals into.
    
    Returns:
        list: A list of hypergraphs representing cascades over time.
    """
    hypergraph_list = []

    # Collect and sort all timestamps
    time_sorted = sorted([time for times in examples_times for time in times[:-1]])
    split_length = len(time_sorted) // step_split  # Number of timestamps in each time slice
    start_time = 0
    end_time = 0

    # Construct hypergraphs for each time slice
    for x in range(split_length, split_length * step_split, split_length):
        start_time = end_time
        end_time = time_sorted[x]

        selected_examples = []
        for i in range(len(examples)):
            example = torch.tensor(examples[i]) if isinstance(examples[i], list) else examples[i]
            example_times = torch.tensor(examples_times[i], dtype=torch.float64) if isinstance(examples_times[i], list) else examples_times[i]

            # Select users participating in the cascade within the current time slice
            selected_example = torch.where((example_times < end_time) & (example_times > start_time),
                                           example, torch.zeros_like(example))
            selected_examples.append(selected_example.numpy().tolist())

        hypergraph_list.append(build_cascade_hypergraph(selected_examples, user_size, device=device))

    # Construct the final hypergraph for the remaining time slice
    selected_examples = []
    for i in range(len(examples)):
        example = torch.tensor(examples[i]) if isinstance(examples[i], list) else examples[i]
        example_times = torch.tensor(examples_times[i], dtype=torch.float64) if isinstance(examples_times[i], list) else examples_times[i]

        # Select users participating after the last time slice
        selected_example = torch.where(example_times > start_time, example, torch.zeros_like(example))
        selected_examples.append(selected_example.numpy().tolist())

    hypergraph_list.append(build_cascade_hypergraph(selected_examples, user_size, device=device))

    return hypergraph_list
