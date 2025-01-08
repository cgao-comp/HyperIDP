import torch
import torch.nn as nn
import torch.nn.functional as F
from Libs.Models.cell import LA_OPS, NA_OPS, POOL_OPS, SC_OPS
from Libs.Models.genotype import (LA_PRIMITIVES, NA_PRIMITIVES,
                                  POOL_PRIMITIVES, SC_PRIMITIVES)


def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")


class NaMixedOp(nn.Module):

    def __init__(self, in_dim, out_dim, with_linear):
        super(NaMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.with_linear = with_linear

        for primitive in NA_PRIMITIVES:
            op = NA_OPS[primitive](in_dim, out_dim)
            self._ops.append(op)

            if with_linear:
                self._ops_linear = nn.ModuleList()
                op_linear = torch.nn.Linear(in_dim, out_dim)
                self._ops_linear.append(op_linear)

    def forward(self, x, weights, edge_index, edge_weight):
        mixed_res = []
        if self.with_linear:
            mixed_res.extend(
                w * F.elu(op(x, edge_index, edge_weight) + linear(x))
                for w, op, linear in zip(weights, self._ops, self._ops_linear)
            )
        else:
            mixed_res.extend(
                w * F.elu(op(x, edge_index, edge_weight)) for w, op in zip(weights, self._ops)
            )
        return sum(mixed_res)


class ScMixedOp(nn.Module):

    def __init__(self):
        super(ScMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in SC_PRIMITIVES:
            op = SC_OPS[primitive]()
            self._ops.append(op)

    def forward(self, x, weights):
        mixed_res = [w * op(x) for w, op in zip(weights, self._ops)]
        return sum(mixed_res)


class LaMixedOp(nn.Module):

    def __init__(self, hidden_size, num_layers=None):
        super(LaMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in LA_PRIMITIVES:
            op = LA_OPS[primitive](hidden_size, num_layers)
            self._ops.append(op)

    def forward(self, x, weights):
        mixed_res = [w * F.relu(op(x)) for w, op in zip(weights, self._ops)]
        return sum(mixed_res)


class PoolingMixedOp(nn.Module):

    def __init__(self, ):
        super(PoolingMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in POOL_PRIMITIVES:
            op = POOL_OPS[primitive]()
            self._ops.append(op)

    def forward(self, x, batch, weights):
        mixed_res = [w * F.relu(op(x, batch))
                     for w, op in zip(weights, self._ops)]
        return sum(mixed_res)
