import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from Libs.Models.genotype import (LA_PRIMITIVES, NA_PRIMITIVES,
                                  POOL_PRIMITIVES, SC_PRIMITIVES)
from Libs.Models.mixed_ops import (LaMixedOp, NaMixedOp, PoolingMixedOp,
                                   ScMixedOp)
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):

    def __init__(self, criterion, in_dim, node_hinfo_dim, edge_dim, out_dim, hidden_size, num_layers=3, dropout=0.5,
                 epsilon=0.0, with_conv_linear=False, use_h_info=False, use_diff_g=False, args=None):
        super(Encoder, self).__init__()
        self.in_dim = in_dim + node_hinfo_dim if use_h_info else in_dim
        self.node_hinfo_dim = node_hinfo_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._criterion = criterion
        self.dropout = dropout
        self.epsilon = epsilon
        # self.explore_num = 0
        self.with_linear = with_conv_linear
        self.use_h_info = use_h_info
        self.use_diff_g = use_diff_g
        self.args = args

        # node aggregator op
        self.lin1 = nn.Linear(self.in_dim, self.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(
                NaMixedOp(self.hidden_size, self.hidden_size, self.with_linear))
        # skip op
        self.scops = nn.ModuleList()
        for _ in range(self.num_layers-1):
            self.scops.append(ScMixedOp())
        if not self.args.fix_last:
            self.scops.append(ScMixedOp())
        # layer aggregator op
        self.laop = LaMixedOp(self.hidden_size, self.num_layers)
        # pooling layer
        self.poolop = PoolingMixedOp()

        self._initialize_alphas()

    def _initialize_alphas(self):
        num_na_ops = len(NA_PRIMITIVES)
        num_sc_ops = len(SC_PRIMITIVES)
        num_la_ops = len(LA_PRIMITIVES)
        num_pool_ops = len(POOL_PRIMITIVES)

        self.na_alphas = Variable(
            1e-3*torch.randn(self.num_layers, num_na_ops).to(device), requires_grad=True)
        if self.args.fix_last:
            self.sc_alphas = Variable(
                1e-3*torch.randn(self.num_layers-1, num_sc_ops).to(device), requires_grad=True)
        else:
            self.sc_alphas = Variable(
                1e-3*torch.randn(self.num_layers, num_sc_ops).to(device), requires_grad=True)
        self.la_alphas = Variable(
            1e-3*torch.randn(1, num_la_ops).to(device), requires_grad=True)
        self.pool_alphas = Variable(
            1e-3*torch.randn(1, num_pool_ops).to(device), requires_grad=True)

        self._arch_parameters = [
            self.na_alphas,
            self.sc_alphas,
            self.la_alphas,
            self.pool_alphas,
        ]

    def forward(self, data):
        # unpack data
        x, batch, edge_index, edge_weight, h_info_node = data.x, data.batch, data.edge_index, data.edge_weight, data.h_info_node
        # generate weights by softmax
        self.na_weights = F.softmax(self.na_alphas, dim=-1)
        self.sc_weights = F.softmax(self.sc_alphas, dim=-1)
        self.la_weights = F.softmax(self.la_alphas, dim=-1)
        self.pool_weights = F.softmax(self.pool_alphas, dim=-1)
        x = torch.cat((x, h_info_node), dim=1) if self.use_h_info else x
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        jk = []
        for i in range(self.num_layers):
            x = self.layers[i](x, self.na_weights[0], edge_index, edge_weight)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.args.fix_last and i == self.num_layers-1:
                jk += [x]
            else:
                jk += [self.scops[i](x, self.sc_weights[i])]

        merge_feature = self.laop(jk, self.la_weights[0])
        merge_feature = F.dropout(
            merge_feature, p=self.dropout, training=self.training)
        # merge_feature = scatter_mean(merge_feature, batch, dim=0)
        readout = self.poolop(merge_feature, batch, self.pool_weights[0])
        return readout


class Classifier(nn.Module):
    def __init__(self, criterion, in_dim, node_hinfo_dim, edge_dim, out_dim, hidden_size, num_layers=3, dropout=0.5,
                 epsilon=0.0, with_conv_linear=False, use_h_info=False, use_diff_g=False, args=None):
        super(Classifier, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self._criterion = criterion
        self.dropout = dropout
        self.epsilon = epsilon
        # self.explore_num = 0
        self.with_linear = with_conv_linear
        self.args = args
        self.encoder = Encoder(criterion, in_dim, node_hinfo_dim, edge_dim, out_dim, hidden_size,
                               num_layers, dropout, epsilon, with_conv_linear, use_h_info, use_diff_g, args)
        self.lin = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim),
            nn.ReLU(),)
        self._initialize_alphas()

    def forward(self, data):
        h = self.encoder(data)
        h = self.lin(h)
        return h

    def _loss(self, data, is_valid=True):
        input = self(data).to(device)
        target = data.y.to(device)
        return self._criterion(input, target)

    def _initialize_alphas(self):

        self._arch_parameters = [
            self.encoder.na_alphas,
            self.encoder.sc_alphas,
            self.encoder.la_alphas,
            self.encoder.pool_alphas
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(na_weights, sc_weights, la_weights, pool_weights):
            gene = []
            na_indices = torch.argmax(na_weights, dim=-1)
            for k in na_indices:
                gene.append(NA_PRIMITIVES[k])
            # sc_indices = sc_weights.argmax(dim=-1)
            sc_indices = torch.argmax(sc_weights, dim=-1)
            for k in sc_indices:
                gene.append(SC_PRIMITIVES[k])
            # la_indices = la_weights.argmax(dim=-1)
            la_indices = torch.argmax(la_weights, dim=-1)
            for k in la_indices:
                gene.append(LA_PRIMITIVES[k])

            pool_indices = torch.argmax(pool_weights, dim=-1)
            for k in pool_indices:
                gene.append(POOL_PRIMITIVES[k])
            return '||'.join(gene)

        gene = _parse(
            F.softmax(self.encoder.na_alphas, dim=-1).data.cpu(),
            F.softmax(self.encoder.sc_alphas, dim=-1).data.cpu(),
            F.softmax(self.encoder.la_alphas, dim=-1).data.cpu(),
            F.softmax(self.encoder.pool_alphas, dim=-1).data.cpu()
        )
        return gene

    def sample_arch(self):
        gene = []
        for _ in range(2):
            for _ in range(3):
                op = np.random.choice(NA_PRIMITIVES, 1)[0]
                gene.append(op)
            for _ in range(2):
                op = np.random.choice(SC_PRIMITIVES, 1)[0]
                gene.append(op)
            op = np.random.choice(LA_PRIMITIVES, 1)[0]
            op = np.random.choice(POOL_PRIMITIVES, 1)[0]
        gene.append(op)
        return '||'.join(gene)


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.args = args
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, data, eta, network_optimizer):
        loss = self.model._loss(data, is_valid=False)  # train loss
        theta = _concat(self.model.parameters()).data  # w
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer']
                             for v in self.model.parameters()).mul_(self.network_momentum)
        except Exception:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(
            loss, self.model.parameters())).data + self.network_weight_decay * theta
        # gradient, L2 norm
        unrolled_model = self._construct_model_from_theta(
            theta.sub(moment + dtheta, alpha=eta))  # one-step update, get w' for Eq.7 in the paper
        return unrolled_model

    def step(self, data, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(data, eta, network_optimizer)
        else:
            self._backward_step(data, is_valid=True)
        self.optimizer.step()

    def _backward_step(self, loader, is_valid=True):
        for data in loader:
            data = data.to(device)
            loss = self.model._loss(data, is_valid)
            loss.backward()

    def _backward_step_unrolled(self, data, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(
            data, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(
            data, is_valid=True)  # validation loss

        unrolled_loss.backward()  # one-step update for w?
        # L_vali w.r.t alpha
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        # gradient, L_train w.r.t w, double check the model construction
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, data)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        # update alpha, which is the ultimate goal of this func, also the goal of the second-order darts
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.to(device)

    def _hessian_vector_product(self, vector, data, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)  # R * d(L_val/w', i.e., get w^+
        loss = self.model._loss(data, is_valid=False)  # train loss
        grads_p = torch.autograd.grad(
            loss, self.model.arch_parameters())  # d(L_train)/d_alpha, w^+

        for p, v in zip(self.model.parameters(), vector):
            # get w^-, need to subtract 2 * R since it has add R
            p.data.sub_(2*R, v)
        loss = self.model._loss(data, is_valid=False)  # train loss
        grads_n = torch.autograd.grad(
            loss, self.model.arch_parameters())  # d(L_train)/d_alpha, w^-

        # reset to the orignial w, always using the self.model, i.e., the original model
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
