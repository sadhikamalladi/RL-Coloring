import torch
import networkx as nx
import numpy as np
from torch.distributions import Normal
from torch.nn import Module, Parameter, Linear, MSELoss
from torch import matmul as mm
from torch.optim import Adam

def init_weights(m):
    if type(m) == Linear:
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)

class S2V(Module):

    def __init__(self, hps):
        # save dimensions
        self.reg_hidden = hps.reg_hidden
        self.embed_dim = hps.embed_dim
        self.pre_pooling = hps.pre_pooling
        self.post_pooling = hps.post_pooling
        self.msg_pass = hps.msg_pass

        # initialize parameters
        init_dist = Normal(loc=torch.tensor([0.0]), scale=torch.tensor([0.01]))
        self.mu_1 = Parameter(init_dist.sample(torch.Size([1, self.embed_dim])))
        self.mu_2 = Linear(self.embed_dim, self.embed_dim, bias=True)

        self.pre_pool = []
        for i in range(self.pre_pooling):
            pre_lin = Linear(self.embed_dim, self.embed_dim, bias=True)
            self.pre_pool.append(pre_lin)

        self.post_pool = []
        for i in range(self.post_pooling):
            post_lin = Linear(self.embed_dim, self.embed_dim, bias=True)
            self.post_pool.append(post_lin)

        self.q_1 = Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_2 = Linear(self.embed_dim, self.embed_dim, bias=True)

        if self.reg_hidden > 0:
            self.q_reg = Linear(2*self.embed_dim, self.reg_hidden)
            self.q = Linear(self.reg_hidden, 1)
        else:
            self.q = Linear(2*self.embed_dim, 1)

        self.apply(init_weights)

        # TODO: add functionality to resume training

    def forward(self, xv, adj):
        bs = xv.shape[0]
        nodes = xv.shape[1]

        for t in range(self.msg_pass):
            if t == 0:
                mu = mm(xv, self.mu_1).clamp(0)
            else:
                mu_1 = mm(xv, self.mu_1).clamp(0)

                for i in range(self.pre_pooling):
                    mu = self.pre_pool[i](mu).clamp(0)

                mu_pool = mm(adj, mu)

                for i in range(self.post_pooling):
                    mu_pool = self.post_pool[i](mu_pool).clamp(0)

                mu_2 = self.mu_2(mu_pool)
                mu = torch.add(mu_1, mu_2).clamp(0)

        q_1 = self.q_1(mm(xv.transpose(1,2), mu))
        q_1 = q_1.expand(bs, nodes, self.embed_dim)
        q_2 = self.q_2(mu)

        q = torch.cat((q_1,q_2), dim=-1)
        if self.reg_hidden > 0:
            q_reg = self.q_reg(q).clamp(0)
            q = self.q(q_reg)
        else:
            q = q.clamp(0)
            q = self.q(q)

        return q

class DQA:

    def __init__(self, hps):
        self.model = S2V(hps)
        self.hps = hps

        self.criterion = MSELoss(reduction='sum').cuda()
        self.optimizer = Adam(self.model.parameters(), lr=hps.lr).cuda()

        self.memory = []
        self.memory_n = []

        self.eps = 1.0
        self.eps_min = 0.02
        self.discount_factor = 0.999990

    def act(self, obs, adj):
        if self.eps > np.random.rand():
            pass
        else:
            q_a = self.model(obs, 
