import torch.nn as nn
from Libs.Models.cell import LA_OPS, NA_OPS, POOL_OPS, SC_OPS
from Libs.Models.mixed_ops import act_map


class NaOp(nn.Module):
    def __init__(self, primitive, in_dim, out_dim, act, with_linear=False):
        super(NaOp, self).__init__()
        self._op = NA_OPS[primitive](in_dim, out_dim)
        self.op_linear = nn.Linear(in_dim, out_dim)
        self.act = act_map(act)
        self.with_linear = with_linear

    def forward(self, x, edge_index):
        if self.with_linear:
            return self.act(self._op(x, edge_index)+self.op_linear(x))
        else:
            return self.act(self._op(x, edge_index))

# class NaMLPOp(nn.Module):
#     def __init__(self, primitive, in_dim, out_dim, act):
#         super(NaMLPOp, self).__init__()
#         self._op = NA_MLP_OPS[primitive](in_dim, out_dim)
#         self.act = act_map(act)

#     def forward(self, x, edge_index):
#         return self.act(self._op(x, edge_index))


class ScOp(nn.Module):
    def __init__(self, primitive):
        super(ScOp, self).__init__()
        self._op = SC_OPS[primitive]()

    def forward(self, x):
        return self._op(x)


class LaOp(nn.Module):
    def __init__(self, primitive, hidden_size, act='relu', num_layers=None):
        super(LaOp, self).__init__()
        self._op = LA_OPS[primitive](hidden_size, num_layers)
        self.act = act_map(act)

    def forward(self, x):
        return self.act(self._op(x))


class PoolOp(nn.Module):
    def __init__(self, primitive, act='relu'):
        super(PoolOp, self).__init__()
        self._op = POOL_OPS[primitive]()
        self.act = act_map(act)

    def forward(self, x, batch):
        return self.act(self._op(x, batch))
