import torch
import torch.nn as nn
import torch.nn.functional as F
from Libs.Models.ops import LaOp, NaOp, PoolOp, ScOp
from torch_geometric.nn import LayerNorm


class Encoder4Tune(nn.Module):

    def __init__(self, genotype, criterion, in_dim, out_dim, hidden_size, num_layers=3, in_dropout=0.5, out_dropout=0.5, act='relu', is_mlp=False, args=None):
        super(Encoder4Tune, self).__init__()
        self.genotype = genotype
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.in_dropout = in_dropout
        self.out_dropout = out_dropout
        self._criterion = criterion
        ops = genotype.split('||')
        self.args = args

        # node aggregator op
        self.lin1 = nn.Linear(in_dim, hidden_size)

        self.gnn_layers = nn.ModuleList(
            [NaOp(ops[i], hidden_size, hidden_size, act, with_linear=args.with_linear) for i in range(num_layers)])

        # skip op
        if self.args.fix_last:
            if self.num_layers > 1:
                self.sc_layers = nn.ModuleList(
                    [ScOp(ops[i+num_layers]) for i in range(num_layers - 1)])
            else:
                self.sc_layers = nn.ModuleList([ScOp(ops[num_layers])])
        else:
            # no output conditions.
            skip_op = ops[num_layers:2 * num_layers]
            if skip_op == ['none'] * num_layers:
                skip_op[-1] = 'skip'
                print('skip_op:', skip_op)
            self.sc_layers = nn.ModuleList(
                [ScOp(skip_op[i]) for i in range(num_layers)])

        # layer norm
        self.lns = torch.nn.ModuleList()
        if self.args.with_layernorm:
            for _ in range(num_layers):
                self.lns.append(LayerNorm(hidden_size, affine=True))

        # layer aggregator op
        self.layer6 = LaOp(ops[-2], hidden_size, 'linear', num_layers)

        self.readout = PoolOp(ops[-1])

    def forward(self, data):
        x, batch, edge_index = data.x, data.batch, data.edge_index
        # generate weights by softmax
        h = self.lin1(x)
        h = F.dropout(h, p=self.in_dropout, training=self.training)
        js = []
        for i in range(self.num_layers):
            h = self.gnn_layers[i](h, edge_index)
            if self.args.with_layernorm:
                # layer_norm = nn.LayerNorm(normalized_shape=x.size(), elementwise_affine=False)
                # x = layer_norm(x)
                h = self.lns[i](h)
            h = F.dropout(h, p=self.in_dropout, training=self.training)
            if i == self.num_layers - 1 and self.args.fix_last:
                js.append(h)
            else:
                js.append(self.sc_layers[i](h))
        h = self.layer6(js)
        h = F.dropout(h, p=self.out_dropout, training=self.training)

        h = self.readout(h, batch)
        return h


class Regressor4Tune(nn.Module):
    def __init__(self, genotype, criterion, in_dim, out_dim, hidden_size, num_layers=3,
                 in_dropout=0.5, out_dropout=0.5, act='relu', is_mlp=False, args=None):
        super(Regressor4Tune, self).__init__()
        self.genotype = genotype
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.in_dropout = in_dropout
        self.out_dropout = out_dropout
        self._criterion = criterion
        self.args = args
        self.encoder = Encoder4Tune(genotype, criterion, in_dim, out_dim, hidden_size,
                                    num_layers, in_dropout, out_dropout, act, is_mlp, args)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim))

    def forward(self, data):
        h = self.encoder(data)
        h = self.classifier(h)
        return h
