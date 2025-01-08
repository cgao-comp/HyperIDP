from torch_geometric.nn import GATConv
import inspect
import sys

import torch
import torch.nn.functional as F
import torch_scatter
# from torch_geometric.utils import scatter_
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import (add_remaining_self_loops, add_self_loops,
                                   remove_self_loops, softmax)
from torch_scatter import scatter_add

# from torch_geometric.nn.conv import MessagePassing

special_args = [
    'edge_index', 'edge_index_i', 'edge_index_j', 'size', 'size_i', 'size_j'
]
__size_error_msg__ = ('All tensors which should get mapped to the same source '
                      'or target nodes must be of same size in dimension 0.')

is_python2 = sys.version_info[0] < 3
getargspec = inspect.getargspec if is_python2 else inspect.getfullargspec


def scatter_(name, src, index, dim=0, dim_size=None):
    assert name in ['add', 'mean', 'max']

    if name == 'max':
        op = torch.finfo if torch.is_floating_point(src) else torch.iinfo
        fill_value = op(src.dtype).min
    else:
        fill_value = 0

    op = getattr(torch_scatter, f'scatter_{name}')
    out = op(src, index, dim, None, dim_size)
    if isinstance(out, tuple):
        out = out[0]

    if name == 'max':
        out[out == fill_value] = 0

    return out


class MessagePassing(torch.nn.Module):

    def __init__(self, aggr='add', flow='source_to_target'):
        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max']

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.__message_args__ = getargspec(self.message)[0][1:]
        self.__special_args__ = [(i, arg)
                                 for i, arg in enumerate(self.__message_args__)
                                 if arg in special_args]
        self.__message_args__ = [
            arg for arg in self.__message_args__ if arg not in special_args
        ]
        self.__update_args__ = getargspec(self.update)[0][2:]

    def propagate(self, edge_index, size=None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (Tensor): The indices of a general (sparse) assignment
                matrix with shape :obj:`[N, M]` (can be directed or
                undirected).
            size (list or tuple, optional): The size :obj:`[N, M]` of the
                assignment matrix. If set to :obj:`None`, the size is tried to
                get automatically inferrred. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct messages
                and to update node embeddings.
        """

        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}

        message_args = []
        for arg in self.__message_args__:
            if arg[-2:] in ij:
                tmp = kwargs.get(arg[:-2], None)
                if tmp is not None:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, (tuple, list)):
                        assert len(tmp) == 2
                        if tmp[1 - idx] is not None:
                            if size[1 - idx] is None:
                                size[1 - idx] = tmp[1 - idx].size(0)
                            if size[1 - idx] != tmp[1 - idx].size(0):
                                raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if size[idx] is None:
                        size[idx] = tmp.size(0)
                    if size[idx] != tmp.size(0):
                        raise ValueError(__size_error_msg__)

                    tmp = torch.index_select(tmp, 0, edge_index[idx])
                message_args.append(tmp)
            else:
                message_args.append(kwargs.get(arg, None))

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs['edge_index'] = edge_index
        kwargs['size'] = size

        for (idx, arg) in self.__special_args__:
            if arg[-2:] in ij:
                message_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                message_args.insert(idx, kwargs[arg])

        update_args = [kwargs[arg] for arg in self.__update_args__]

        out = self.message(*message_args)
        if self.aggr in ["add", "mean", "max"]:
            out = scatter_(self.aggr, out, edge_index[i], dim_size=size[i])
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
        for each edge in :math:`(i,j) \in \mathcal{E}`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

        return x_j

    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""

        return aggr_out


class GeoLayer(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 att_type="gat",
                 agg_type="sum",
                 pool_dim=0):
        if agg_type in ["sum", "mlp"]:
            super(GeoLayer, self).__init__('add')
        elif agg_type in ["mean", "max"]:
            super(GeoLayer, self).__init__(agg_type)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.att_type = att_type
        self.agg_type = agg_type

        self.weight = Parameter(torch.Tensor(
            in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if self.att_type in ["generalized_linear"]:
            self.general_att_layer = torch.nn.Linear(
                out_channels, 1, bias=False)
        self.reset_parameters()

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 2 if improved else 1
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

        if self.att_type in ["generalized_linear"]:
            glorot(self.general_att_layer.weight)

        # if self.pool_dim != 0:
        #    for layer in self.pool_layer:
        #        glorot(layer.weight)
        #        zeros(layer.bias)

    def forward(self, x, edge_index, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # prepare
        if torch.is_tensor(x):
            x = torch.mm(x, self.weight).view(-1,
                                              self.heads, self.out_channels)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight).view(-1, self.heads, self.out_channels),
                 None if x[1] is None else torch.matmul(x[1], self.weight).view(-1, self.heads, self.out_channels))
        num_nodes = x.size(0) if torch.is_tensor(x) else size[0]
        return self.propagate(edge_index, size=size, x=x, num_nodes=num_nodes)

    def message(self, x_i, x_j, edge_index, num_nodes):

        if self.att_type == "const":
            if self.training and self.dropout > 0:
                x_j = F.dropout(x_j, p=self.dropout, training=True)
            return x_j
        else:
            # Compute attention coefficients.
            alpha = self.apply_attention(edge_index, num_nodes, x_i, x_j)
            alpha = softmax(
                alpha, edge_index[0], ptr=None, num_nodes=num_nodes)
            # Sample attention coefficients stochastically.
            if self.training and self.dropout > 0:
                alpha = F.dropout(alpha, p=self.dropout, training=True)

            return x_j * alpha.view(-1, self.heads, 1)

    def apply_attention(self, edge_index, num_nodes, x_i, x_j):
        if self.att_type == "gat":
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)

        elif self.att_type == "gat_sym":
            wl = self.att[:, :, :self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels:]  # weight right
            alpha = (x_i * wl).sum(dim=-1) + (x_j * wr).sum(dim=-1)
            alpha_2 = (x_j * wl).sum(dim=-1) + (x_i * wr).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope) + \
                F.leaky_relu(alpha_2, self.negative_slope)

        elif self.att_type == "linear":
            alpha = self._extracted_from_apply_attention_15(x_j)
        elif self.att_type == "cos":
            wl = self.att[:, :, :self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels:]  # weight right
            alpha = x_i * wl * x_j * wr
            alpha = alpha.sum(dim=-1)

        elif self.att_type == "generalized_linear":
            alpha = self._extracted_from_apply_attention_28(x_i, x_j)
        else:
            raise Exception("Wrong attention type:", self.att_type)
        return alpha

    # TODO Rename this here and in `apply_attention`
    def _extracted_from_apply_attention_28(self, x_i, x_j):
        wl = self.att[:, :, :self.out_channels]  # weight left
        wr = self.att[:, :, self.out_channels:]  # weight right
        al = x_i * wl
        ar = x_j * wr
        result = al + ar
        result = torch.tanh(result)
        result = self.general_att_layer(result)
        return result

    # TODO Rename this here and in `apply_attention`
    def _extracted_from_apply_attention_15(self, x_j):
        wl = self.att[:, :, :self.out_channels]  # weight left
        wr = self.att[:, :, self.out_channels:]  # weight right
        al = x_j * wl
        ar = x_j * wr
        result = al.sum(dim=-1) + ar.sum(dim=-1)
        result = torch.tanh(result)
        return result

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})'


dim = 256
lstm_hidden = 256
layer_num = 4


class Breadth(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Breadth, self).__init__()
        self.gatconv = GATConv(in_dim, out_dim, heads=1)

    def forward(self, x, edge_index, size=None):
        x = torch.tanh(self.gatconv(x, edge_index, size=size))
        return x


class Depth(torch.nn.Module):

    def __init__(self, in_dim, hidden):
        super(Depth, self).__init__()
        self.lstm = torch.nn.LSTM(in_dim, hidden, 1, bias=False)

    def forward(self, x, h, c):
        x, (h, c) = self.lstm(x, (h, c))
        return x, (h, c)


class GeniePathLayer(torch.nn.Module):

    def __init__(self, in_dim, hidden):
        super(GeniePathLayer, self).__init__()
        self.breadth_func = Breadth(in_dim, hidden)
        self.depth_func = Depth(hidden, hidden)
        self.in_dim = in_dim
        self.hidden = hidden
        self.lstm_hidden = hidden

    def forward(self, x, edge_index, size=None):
        if torch.is_tensor(x):
            h = torch.zeros(1, x.shape[0], self.lstm_hidden, device=x.device)
            c = torch.zeros(1, x.shape[0], self.lstm_hidden, device=x.device)
        else:
            h = torch.zeros(1, x[1].shape[0],
                            self.lstm_hidden, device=x[1].device)
            c = torch.zeros(1, x[1].shape[0],
                            self.lstm_hidden, device=x[1].device)
        x = self.breadth_func(x, edge_index, size=size)
        x = x[None, :]
        x, (h, c) = self.depth_func(x, h, c)
        x = x[0]
        return x


class GeniePath(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super(GeniePath, self).__init__()
        self.lin1 = torch.nn.Linear(in_dim, dim)
        self.gplayers = torch.nn.ModuleList(
            [GeniePathLayer(dim) for _ in range(layer_num)]
        )
        self.lin2 = torch.nn.Linear(dim, out_dim)

    def forward(self, x, edge_index, size=None):
        x = self.lin1(x)
        h = torch.zeros(1, x.shape[0], lstm_hidden, device=x.device)
        c = torch.zeros(1, x.shape[0], lstm_hidden, device=x.device)
        for i, l in enumerate(self.gplayers):
            x, (h, c) = self.gplayers[i](x, edge_index, h, c)
        x = self.lin2(x)
        return x


class GeniePathLazy(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super(GeniePathLazy, self).__init__()
        self.lin1 = torch.nn.Linear(in_dim, dim)
        self.breadths = torch.nn.ModuleList(
            [Breadth(dim, dim) for _ in range(layer_num)]
        )
        self.depths = torch.nn.ModuleList(
            [Depth(dim * 2, lstm_hidden) for _ in range(layer_num)]
        )
        self.lin2 = torch.nn.Linear(dim, out_dim)

    def forward(self, x, edge_index):
        x = self.lin1(x)
        h = torch.zeros(1, x.shape[0], lstm_hidden, device=x.device)
        c = torch.zeros(1, x.shape[0], lstm_hidden, device=x.device)
        h_tmps = [self.breadths[i](x, edge_index)
                  for i, l in enumerate(self.breadths)]
        x = x[None, :]
        for i, l in enumerate(self.depths):
            in_cat = torch.cat((h_tmps[i][None, :], x), -1)
            x, (h, c) = self.depths[i](in_cat, h, c)
        x = self.lin2(x[0])
        return x
