import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import (GATConv, GatedGraphConv, GCNConv, GINEConv,
                                GraphConv, JumpingKnowledge, ResGatedGraphConv,
                                TAGConv)
from torch_geometric.nn.pool import (global_add_pool, global_max_pool,
                                     global_mean_pool)
from torch_geometric_temporal.nn import DCRNN, TGCN, EvolveGCNO

# Node aggregator
NA_OPS = {
    'gcn': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gcn'),
    'gat': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gat'),
    'gconv_add': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'add'),
    'gconv_mean': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'mean'),
    'gconv_max': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'max'),

    'gated': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gated'),
    'res_gated': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'res_gated'),
    'tag': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'tag'),
    'gine': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'gine'),

    'dcrnn': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'dcrnn'),
    'tgcn': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'tgcn'),
    'egcn': lambda in_dim, out_dim: NaAggregator(in_dim, out_dim, 'egcn'),
}

# Layer aggregator
LA_OPS = {
    'l_max': lambda hidden_size, num_layers: LaAggregator('max', hidden_size, num_layers),
    'l_concat': lambda hidden_size, num_layers: LaAggregator('cat', hidden_size, num_layers),
    'l_lstm': lambda hidden_size, num_layers: LaAggregator('lstm', hidden_size, num_layers),
    'l_sum': lambda hidden_size, num_layers: LaAggregator('sum', hidden_size, num_layers),
    'l_att': lambda hidden_size, num_layers: LaAggregator('att', hidden_size, num_layers),
    'l_mean': lambda hidden_size, num_layers: LaAggregator('mean', hidden_size, num_layers)
}

# Skip-connection
SC_OPS = {
    'none': lambda: Zero(),
    'skip': lambda: Identity(),
}

# Pooling
POOL_OPS = {
    'p_max': lambda: PoolingAggregator('max'),
    'p_mean': lambda: PoolingAggregator('mean'),
    'p_add': lambda: PoolingAggregator('add'),
}


class NaAggregator(nn.Module):

    def __init__(self, in_dim, out_dim, aggregator):
        super(NaAggregator, self).__init__()
        if aggregator == 'gcn':
            self._op = GCNConv(in_dim, out_dim)
        if aggregator == 'gat':
            heads = 8
            out_dim //= heads
            self._op = GATConv(in_dim, out_dim, heads=heads, dropout=0.5)
        if aggregator in ['add', 'mean', 'max']:
            self._op = GraphConv(in_dim, out_dim, aggr=aggregator)
        if aggregator == 'gated':
            num_layers = 3
            self._op = GatedGraphConv(out_dim, num_layers=num_layers)
        if aggregator == 'res_gated':
            self._op = ResGatedGraphConv(in_dim, out_dim)
        if aggregator == 'tag':
            self._op = TAGConv(in_dim, out_dim)
        if aggregator == 'gine':
            _nn = Sequential(Linear(in_dim, out_dim), ReLU(),
                             Linear(out_dim, out_dim))
            edge_dim = 1
            self._op = GINEConv(_nn, edge_dim=edge_dim)
        if aggregator == 'dcrnn':
            K = 1
            self._op = DCRNN(in_dim, out_dim, K=K)
        if aggregator == 'tgcn':
            self._op = TGCN(in_dim, out_dim)
        if aggregator == 'egcn':
            self._op = EvolveGCNO(in_dim)

    def forward(self, x, edge_index, edge_weight):
        return self._op(x, edge_index, edge_weight)


class LaAggregator(nn.Module):

    def __init__(self, mode, hidden_size, num_layers=3):
        super(LaAggregator, self).__init__()
        self.mode = mode
        if mode in ['lstm', 'cat', 'max']:
            self.jump = JumpingKnowledge(
                mode, channels=hidden_size, num_layers=num_layers)
        elif mode == 'att':
            self.att = Linear(hidden_size, 1)

        if mode == 'cat':
            self.lin = Linear(hidden_size * num_layers, hidden_size)
        else:
            self.lin = Linear(hidden_size, hidden_size)

    def forward(self, xs):
        if self.mode in ['lstm', 'cat', 'max']:
            output = self.jump(xs)
        elif self.mode == 'sum':
            output = torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'mean':
            output = torch.stack(xs, dim=-1).mean(dim=-1)
        elif self.mode == 'att':
            input = torch.stack(xs, dim=-1).transpose(1, 2)
            weight = self.att(input)
            weight = F.softmax(weight, dim=1)
            output = torch.mul(input, weight).transpose(1, 2).sum(dim=-1)

        return self.lin(F.relu(output))


class PoolingAggregator(nn.Module):

    def __init__(self, mode):
        super(PoolingAggregator, self).__init__()
        self.mode = mode
        if mode == 'add':
            self.readout = global_add_pool
        elif mode == 'mean':
            self.readout = global_mean_pool
        elif mode == 'max':
            self.readout = global_max_pool

    def forward(self, h, batch):
        return self.readout(h, batch)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x.mul(0.)
