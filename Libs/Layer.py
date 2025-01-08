"""
Module Name: Graph Convolution Layers for Cascade Prediction Models
Developer: hwxu
Development Environment: Python 3.8+, PyTorch
Date: July, 2024
Version: Bean
Description: Implements various graph convolution layers, including HypergraphConv, GCNConv, 
             GraphSAGEConv, GATConv, and LinearLayer, to handle dynamic cascade hypergraph 
             convolution, user embedding through GCN and GraphSAGE, and attention mechanisms 
             for cascade graphs.
"""

import math

import torch
import torch.nn as nn


class HypergraphConv(nn.Module):
    """
    Dynamic cascade hypergraph convolution layer.
    
    Args:
        input_dim (int): Input dimension of node features.
        output_dim (int): Output dimension of node features after transformation.
        bias (bool): Whether to use bias in the linear transformation.
        drop_rate (float): Dropout rate applied after activation.
    
    Forward Args:
        X (torch.Tensor): Input feature matrix of shape (N, input_dim).
        hypergraph: The hypergraph object with methods v2e (vertex to edge) and e2v (edge to vertex).
        
    Returns:
        torch.Tensor: Transformed node feature matrix after hypergraph convolution.
    """
    def __init__(self, input_dim, output_dim, bias=True, drop_rate=0.5):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, X, hypergraph):
        X = self.theta(X)
        Y = hypergraph.v2e(X, aggr="mean")
        X_ = hypergraph.e2v(Y, aggr="mean")
        X_ = self.drop(self.act(X_))
        return X_


class GCNConv(nn.Module):
    """
    Social graph GCN convolution layer for generating user embeddings.
    
    Args:
        input_dim (int): Input dimension of node features.
        output_dim (int): Output dimension of node features after transformation.
        bias (bool): Whether to use bias in the linear transformation.
        drop_rate (float): Dropout rate applied after activation.
    
    Forward Args:
        X (torch.Tensor): Input feature matrix of shape (N, input_dim).
        relationgraph: The relation graph object with a smoothing_with_GCN method.
        
    Returns:
        torch.Tensor: Transformed node feature matrix after GCN convolution.
    """
    def __init__(self, input_dim, output_dim, bias=True, drop_rate=0.5):
        super(GCNConv, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, X, relationgraph):
        X = self.theta(X)
        X_ = relationgraph.smoothing_with_GCN(X)
        X_ = self.drop(self.act(X_))
        return X_


class GraphSAGEConv(nn.Module):
    """
    Social graph GraphSAGE convolution layer for generating user embeddings.
    
    Args:
        input_dim (int): Input dimension of node features.
        output_dim (int): Output dimension of node features after transformation.
        bias (bool): Whether to use bias in the linear transformation.
        drop_rate (float): Dropout rate applied after activation.
    
    Forward Args:
        X (torch.Tensor): Input feature matrix of shape (N, input_dim).
        relationgraph: The relation graph object with a v2v method (neighbor aggregation).
        
    Returns:
        torch.Tensor: Transformed node feature matrix after GraphSAGE convolution.
    """
    def __init__(self, input_dim, output_dim, bias=True, drop_rate=0.5):
        super(GraphSAGEConv, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(input_dim * 2, output_dim, bias=bias)

    def forward(self, X, relationgraph):
        X_nbr = relationgraph.v2v(X, aggr="mean")
        X = torch.cat([X, X_nbr], dim=1)
        X_ = self.theta(X)
        X_ = self.drop(self.act(X_))
        return X_


class GATConv(nn.Module):
    """
    Directed cascade graph GAT (Graph Attention Network) convolution layer.
    
    Args:
        input_dim (int): Input dimension of node features.
        output_dim (int): Output dimension of node features after transformation.
        bias (bool): Whether to use bias in the linear transformation.
        drop_rate (float): Dropout rate applied after attention score computation.
        atten_neg_slope (float): Negative slope of the LeakyReLU used in attention mechanism.
    
    Forward Args:
        X (torch.Tensor): Input feature matrix of shape (N, input_dim).
        cascade_graph: The directed cascade graph object with v2v (vertex to vertex) method.
        
    Returns:
        torch.Tensor: Transformed node feature matrix after GAT convolution.
    """
    def __init__(self, input_dim, output_dim, bias=True, drop_rate=0.5, atten_neg_slope=0.2):
        super().__init__()
        self.atten_dropout = nn.Dropout(drop_rate)
        self.atten_act = nn.LeakyReLU(atten_neg_slope)
        self.act = nn.ELU(inplace=True)
        self.theta = nn.Linear(input_dim, output_dim, bias=bias)
        self.atten_src = nn.Linear(output_dim, 1, bias=False)
        self.atten_dst = nn.Linear(output_dim, 1, bias=False)

    def forward(self, X, cascade_graph):
        X = self.theta(X)
        X_for_src = self.atten_src(X)
        X_for_dst = self.atten_dst(X)
        e_atten_score = X_for_src[cascade_graph.e_src] + X_for_dst[cascade_graph.e_dst]
        e_atten_score = self.atten_dropout(self.act(e_atten_score).squeeze())
        X_ = cascade_graph.v2v(X, aggr='softmax_then_sum', e_weight=e_atten_score)
        X_ = self.act(X_)
        return X_


class LinearLayer(nn.Module):
    """
    Linear transformation layer with optional L2 regularization.
    
    Args:
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        is_regularize (bool): Whether to apply L2 regularization on weights.
    
    Forward Args:
        shared_embedding (torch.Tensor): Input embedding of shape (N, input_dim).
        device (str): Device for computation ('cpu' or 'cuda').
        
    Returns:
        tuple: Output tensor after transformation and the L2 regularization loss.
    """
    def __init__(self, input_dim, output_dim, is_regularize=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_regularize = is_regularize
        self.U = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        self.b = nn.Parameter(torch.Tensor(self.output_dim))
        self.init_weights()

    def init_weights(self):
        """Initialize the weights of the linear layer."""
        stdv = 1.0 / math.sqrt(self.input_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, shared_embedding, device):
        """ 
        Forward pass for the linear layer.
        
        Args:
            shared_embedding (torch.Tensor): Input embedding matrix of shape (N, input_dim).
            device (str): Device for computation ('cpu' or 'cuda').
        
        Returns:
            tuple: Output tensor and L2 regularization loss (if applied).
        """
        loss_l2 = torch.zeros(1, dtype=torch.float32, device=device)
        output = shared_embedding @ self.U + self.b
        if self.is_regularize:
            loss_l2 += (torch.norm(self.U) ** 2) / 2 + (torch.norm(self.b) ** 2) / 2

        return output, loss_l2
