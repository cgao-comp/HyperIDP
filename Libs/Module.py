"""
Module Name: Graph-Based Cascade Prediction Models
Developer: hwxu
Development Environment: Python 3.8+, PyTorch, NumPy
Date: August, 2024
Version: Bean
Description: This module defines neural network models for cascade prediction using social 
             and dynamic hypergraph-based methods. It includes implementations of GNNs, 
             LSTMs, and HGNNs to learn user embeddings from both social relation graphs 
             and dynamic hypergraphs. The model combines these embeddings for cascade 
             prediction, along with adversarial and differential loss mechanisms to 
             improve generalization and task performance.
"""


import math

import torch
import torch.nn as nn
import torch.nn.init as init
from Layer import *


class RelationGNN(nn.Module):
    """
    Relation Graph Neural Network (GNN) for learning user embeddings from a social graph.
    
    Args:
        input_num (int): Number of users.
        embed_dim (int): Dimension of the embeddings.
        dropout (float): Dropout rate for regularization.
        is_norm (bool): Whether to apply batch normalization.
    """
    def __init__(self, input_num, embed_dim, dropout=0.5, is_norm=False):
        super().__init__()
        self.user_embedding = nn.Embedding(input_num, embed_dim)
        self.graphsage = GraphSAGEConv(embed_dim, embed_dim)
        self.is_norm = is_norm
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
        if self.is_norm:
            self.batch_norm = nn.BatchNorm1d(embed_dim)
        self.init_weights()

    def init_weights(self):
        """Initializes the user embeddings using Xavier normal initialization."""
        init.xavier_normal_(self.user_embedding.weight)

    def forward(self, relation_graph):
        """ 
        Forward pass through the GraphSAGE layer and optional batch normalization.
        
        Args:
            relation_graph: Graph structure for user relations.
        
        Returns:
            torch.Tensor: Learned user embeddings.
        """
        gnn_embeddings = self.graphsage(self.user_embedding.weight, relation_graph)
        gnn_embeddings = self.dropout(gnn_embeddings)
        if self.is_norm:
            gnn_embeddings = self.batch_norm(gnn_embeddings)
        return gnn_embeddings


class Fusion(nn.Module):
    """
    Fusion layer to combine hidden embeddings with dynamic embeddings using attention mechanism.
    
    Args:
        input_size (int): Size of the input embeddings.
        out (int): Output size.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, input_size, out=1, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        """Initializes the weights using Xavier normal initialization."""
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        """
        Combines hidden embeddings and dynamic embeddings with attention-based weighting.
        
        Args:
            hidden (torch.Tensor): Hidden embeddings of shape (user_count, embedding_dim).
            dy_emb (torch.Tensor): Dynamic embeddings of shape (user_count, embedding_dim).
        
        Returns:
            torch.Tensor: Weighted sum of hidden and dynamic embeddings.
        """
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = nn.functional.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        return torch.sum(emb_score * emb, dim=0)


class DynamicCasHGNN(nn.Module):
    """
    Dynamic Hypergraph Neural Network (HGNN) for learning embeddings over a sequence of hypergraphs.
    
    Args:
        input_num (int): Number of users.
        embed_dim (int): Dimension of the embeddings.
        step_split (int): Number of hypergraphs in the sequence.
        dropout (float): Dropout rate for regularization.
        is_norm (bool): Whether to apply batch normalization.
    """
    def __init__(self, input_num, embed_dim, step_split=8, dropout=0.5, is_norm=False):
        super().__init__()
        self.input_num = input_num
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.is_norm = is_norm
        self.step_split = step_split
        if self.is_norm:
            self.batch_norm = nn.BatchNorm1d(embed_dim)
        self.user_embeddings = nn.Embedding(input_num, embed_dim)
        self.hgnn = HypergraphConv(embed_dim, embed_dim, drop_rate=dropout)
        self.fus = Fusion(embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the user embeddings using Xavier normal initialization."""
        init.xavier_normal_(self.user_embeddings.weight)

    def forward(self, hypergraph_list, device=torch.device('cuda')):
        """
        Forward pass through the sequence of hypergraphs, combining learned embeddings over time.
        
        Args:
            hypergraph_list (list): List of hypergraphs in the sequence.
            device (torch.device): Device for computation (default: 'cuda').
        
        Returns:
            torch.Tensor: Final user embeddings after hypergraph convolution.
        """
        hg_embeddings = []
        for i, hg in enumerate(hypergraph_list):
            subhg_embedding = self.hgnn(self.user_embeddings.weight, hg)
            if i > 0:
                subhg_embedding = self.fus(hg_embeddings[-1], subhg_embedding)
            hg_embeddings.append(subhg_embedding)
        return hg_embeddings[-1]


class RelationLSTM(nn.Module):
    """
    LSTM to process user embeddings learned from a social graph.
    
    Args:
        embed_dim (int): Dimension of the embeddings.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers=1, batch_first=True)

    def lookup_embedding(self, examples, embeddings):
        """
        Looks up the user embeddings for a batch of examples.
        
        Args:
            examples (torch.Tensor): Batch of user sequences.
            embeddings (torch.Tensor): Learned user embeddings.
        
        Returns:
            torch.Tensor: Embeddings for the given batch.
        """
        return torch.stack([torch.index_select(embeddings, dim=0, index=example) for example in examples], 0)

    def forward(self, examples, user_social_embedding):
        """
        Forward pass through the LSTM layer.
        
        Args:
            examples (torch.Tensor): User sequences (batch_size, sequence_length).
            user_social_embedding (torch.Tensor): User embeddings from the social graph.
        
        Returns:
            torch.Tensor: Output embeddings from the LSTM.
        """
        user_embedding = self.lookup_embedding(examples, user_social_embedding)
        output_embedding, _ = self.lstm(user_embedding)
        return output_embedding


class CascadeLSTM(nn.Module):
    """
    LSTM to process user embeddings learned from a dynamic cascade graph.
    
    Args:
        emb_dim (int): Dimension of the embeddings.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.lstm = nn.LSTM(emb_dim, emb_dim, num_layers=1, batch_first=True)

    def lookup_embedding(self, examples, embeddings):
        """
        Looks up the user embeddings for a batch of examples.
        
        Args:
            examples (torch.Tensor): Batch of user sequences.
            embeddings (torch.Tensor): Learned user embeddings.
        
        Returns:
            torch.Tensor: Embeddings for the given batch.
        """
        return torch.stack([torch.index_select(embeddings, dim=0, index=example) for example in examples], 0)

    def forward(self, examples, user_cas_embedding):
        """
        Forward pass through the LSTM layer.
        
        Args:
            examples (torch.Tensor): User sequences (batch_size, sequence_length).
            user_cas_embedding (torch.Tensor): User embeddings from the dynamic cascade graph.
        
        Returns:
            torch.Tensor: Output embeddings from the LSTM.
        """
        cas_embedding = self.lookup_embedding(examples, user_cas_embedding)
        output_embedding, _ = self.lstm(cas_embedding)
        return output_embedding


class SharedLSTM(nn.Module):
    """
    Shared LSTM for processing both cascade and social graph embeddings.
    
    Args:
        input_size (int): Sequence length of user embeddings.
        emb_dim (int): Dimension of the embeddings.
    """
    def __init__(self, input_size, emb_dim):
        super().__init__()
        self.input_size = input_size
        self.emb_dim = emb_dim

        # LSTM parameters for cascade, social, and hidden embeddings
        self.W_i = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.U_i = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.V_i = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.b_i = nn.Parameter(torch.Tensor(emb_dim))

        self.W_f = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.U_f = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.V_f = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.b_f = nn.Parameter(torch.Tensor(emb_dim))

        self.W_c = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.U_c = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.V_c = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.b_c = nn.Parameter(torch.Tensor(emb_dim))

        self.W_o = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.U_o = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.V_o = nn.Parameter(torch.Tensor(emb_dim, emb_dim))
        self.b_o = nn.Parameter(torch.Tensor(emb_dim))

        self.init_weights()

    def init_weights(self):
        """Initializes the LSTM parameters using uniform distribution."""
        stdv = 1.0 / math.sqrt(self.emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, cas_emb, social_emb, init_states=None):
        """
        Forward pass through the shared LSTM for combining cascade and social embeddings.
        
        Args:
            cas_emb (torch.Tensor): User embeddings from the cascade graph (batch_size, seq_len, emb_dim).
            social_emb (torch.Tensor): User embeddings from the social graph (batch_size, seq_len, emb_dim).
            init_states (tuple): Initial hidden and cell states for the LSTM (optional).
        
        Returns:
            tuple: (hidden sequence, final hidden and cell states).
        """
        bs, seq_sz, _ = cas_emb.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.emb_dim).to(cas_emb.device),
                torch.zeros(bs, self.emb_dim).to(cas_emb.device)
            )
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):
            cas_emb_t = cas_emb[:, t, :]
            social_emb_t = social_emb[:, t, :]

            i_t = torch.sigmoid(cas_emb_t @ self.W_i + social_emb_t @ self.U_i + h_t @ self.V_i + self.b_i)
            f_t = torch.sigmoid(cas_emb_t @ self.W_f + social_emb_t @ self.U_f + h_t @ self.V_f + self.b_f)
            g_t = torch.tanh(cas_emb_t @ self.W_c + social_emb_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t = torch.sigmoid(cas_emb_t @ self.W_o + social_emb_t @ self.U_o + h_t @ self.V_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) for prediction tasks.
    
    Args:
        input_dim (int): Input dimension size.
        hidden_dim (int): Hidden layer dimension size.
        output_dim (int): Output layer dimension size.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        """Initializes the MLP weights using uniform distribution."""
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, X):
        """
        Forward pass through the MLP layers.
        
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        out = self.relu1(self.linear1(X))
        out = self.relu2(self.linear2(out))
        return self.linear3(out)


class Module(nn.Module):
    """
    Main model module that combines different components of the pipeline.
    
    Args:
        user_size (int): Number of users in the dataset.
        embed_dim (int): Dimension of the user embeddings.
        step_split (int): Number of hypergraphs in the sequence.
        max_seq_len (int): Maximum sequence length for cascades.
        task_num (int): Number of tasks for multi-task learning.
        device (torch.device): Device for computation.
    """
    def __init__(self, user_size, embed_dim, step_split=8, max_seq_len=200, task_num=2, device=torch.device('cuda')):
        super().__init__()
        self.user_size = user_size
        self.emb_dim = embed_dim
        self.step_split = step_split
        self.max_seq_len = max_seq_len
        self.device = device
        self.task_num = task_num
        self.task_label = torch.LongTensor([i for i in range(self.task_num)])
        self.dycasHGNN = DynamicCasHGNN(self.user_size, self.emb_dim, self.step_split)
        self.relationGNN = RelationGNN(self.user_size, self.emb_dim)
        self.relationLSTM = RelationLSTM(self.emb_dim)
        self.cascadeLSTM = CascadeLSTM(self.emb_dim)
        self.sharedLSTM = SharedLSTM(self.max_seq_len, self.emb_dim)
        self.shared_linear = LinearLayer(self.emb_dim, self.task_num)
        self.micro_mlp = MLP(self.emb_dim * 2, self.emb_dim * 4, self.user_size)
        self.macro_mlp = MLP(self.emb_dim * 2, self.emb_dim * 4, 1)
        self.user_embedding = nn.Embedding(self.user_size, self.emb_dim)
        self.init_weights()

    def init_weights(self):
        """Initializes the embeddings using Xavier normal initialization."""
        stdv = 1.0 / math.sqrt(self.emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def lookup_embedding(self, examples, embeddings):
        """
        Looks up the user embeddings for a batch of examples.
        
        Args:
            examples (torch.Tensor): Batch of user sequences.
            embeddings (torch.Tensor): Learned user embeddings.
        
        Returns:
            torch.Tensor: Embeddings for the given batch.
        """
        return torch.stack([torch.index_select(embeddings, dim=0, index=example) for example in examples], 0)

    def adversarial_loss(self, shared_embedding):
        """
        Computes adversarial loss using a shared linear layer.
        
        Args:
            shared_embedding (torch.Tensor): Shared embeddings for multiple tasks.
        
        Returns:
            tuple: (loss_adv, loss_l2) adversarial loss and L2 regularization loss.
        """
        logits, loss_l2 = self.shared_linear(shared_embedding, self.device)
        loss_adv = torch.zeros(logits.shape[0], device=self.device)
        for task in range(self.task_num):
            label = torch.tensor([task] * logits.shape[0]).to(self.device)
            loss_adv += torch.nn.CrossEntropyLoss(reduce=False)(logits, label.long())
        return torch.mean(loss_adv), loss_l2

    def diff_loss(self, shared_embedding, task_embedding):
        """
        Computes difference loss between shared and task-specific embeddings to ensure diversity.
        
        Args:
            shared_embedding (torch.Tensor): Shared embeddings.
            task_embedding (torch.Tensor): Task-specific embeddings.
        
        Returns:
            torch.Tensor: Difference loss.
        """
        shared_embedding -= torch.mean(shared_embedding, 0)
        task_embedding -= torch.mean(task_embedding, 0)
        shared_embedding = nn.functional.normalize(shared_embedding, dim=1, p=2)
        task_embedding = nn.functional.normalize(task_embedding, dim=1, p=2)
        correlation_matrix = task_embedding.t() @ shared_embedding
        loss_diff = torch.mean(torch.square(correlation_matrix)) * 0.01
        return torch.clamp(loss_diff, min=0)

    def forward(self, graph_list, relation_graph, examples):
        """
        Forward pass through the entire model pipeline.
        
        Args:
            graph_list (list): List of hypergraphs.
            relation_graph: Social relation graph.
            examples (torch.Tensor): User sequences.
        
        Returns:
            tuple: (pred_micro, pred_macro, loss_adv, loss_diff) predictions and losses.
        """
        user_cas_embedding = self.dycasHGNN(graph_list, self.device)
        user_social_embedding = self.relationGNN(relation_graph)
        sender_social_embedding = self.relationLSTM(examples, user_social_embedding)
        sender_cas_embedding = self.cascadeLSTM(examples, user_cas_embedding)
        sender_cas_embedding_share = self.lookup_embedding(examples, user_cas_embedding)
        sender_social_embedding_share = self.lookup_embedding(examples, user_social_embedding)
        shared_embedding, _ = self.sharedLSTM(sender_cas_embedding_share, sender_social_embedding_share)

        example_len = torch.count_nonzero(examples, 1)
        batch_size, seq_len, emb_dim = shared_embedding.size()
        H_user, H_cas, H_share = [], [], []

        for i in range(batch_size):
            H_user.append(sender_social_embedding[i, example_len[i] - 1, :])
            H_cas.append(sender_cas_embedding[i, example_len[i] - 1, :])
            H_share.append(shared_embedding[i, example_len[i] - 1, :])

        H_user = torch.stack(H_user, dim=0)
        H_cas = torch.stack(H_cas, dim=0)
        H_share = torch.stack(H_share, dim=0)

        pred_micro = self.micro_mlp(torch.cat((sender_social_embedding, shared_embedding), dim=2))
        pred_macro = self.macro_mlp(torch.cat((H_cas, H_share), dim=1))
        loss_adv, _ = self.adversarial_loss(H_share)
        loss_diff_micro = self.diff_loss(H_share, H_user)
        loss_diff_macro = self.diff_loss(H_share, H_cas)
        loss_diff = loss_diff_micro + loss_diff_macro

        return pred_micro, pred_macro, loss_adv.item(), loss_diff.item()
