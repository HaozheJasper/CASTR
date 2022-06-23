"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F
from math import sqrt
from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm


def identity(x):
    return x


class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.leaky_relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=True,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)

class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class MlpEncoder(FlattenMlp):
    '''
    encode context via MLP
    '''

    def reset(self, num_tasks=1):
        pass

class GraphEncoder(MlpEncoder):
    def __init__(self, hidden_sizes,
                 output_size,
                 mid_size,
                 num_component,
                 input_size,
                 norm_func=F.softmax,
                 mode='1', # 111； [234][01][234]
                 selfloop=True,
                 **kwargs):
        self.save_init_params(locals())
        super(GraphEncoder, self).__init__(
            hidden_sizes=hidden_sizes,
            output_size=mid_size,
            input_size=input_size,
            **kwargs
        )
        self.c = mid_size
        self.d = num_component
        self.normalization_func = norm_func
        self.mode = mode
        self.inp_dim = self.c
        self.selfloop = selfloop
        relu = nn.LeakyReLU

        self.n_head = 1
        self.n_round = 1
        if self.n_head>1:
            self.w3_transforms = nn.ModuleList([nn.Sequential(
                nn.Linear(self.c*self.n_head, self.c),
                relu()
            ) for _ in range(self.n_round)])

        self.dim_red = nn.Linear(self.c, output_size)
        # the number of stages determines whether another agg is used
        num_stage = eval(mode[0])
        # if num_stage>0:# ==0, only agg stage
        self.w_transform = nn.Sequential( # either use relu or use norm
            nn.Linear(self.c, 1),
            # LayerNorm(self.d),
            relu()
        )
        if num_stage>1: # 3stage
            self.w2_transform = nn.Sequential( # either use relu or use norm
                nn.Linear(self.c, self.d),  # c,d
                # LayerNorm(self.d),
                relu() #
            )
        # the style of aggregation
        if mode[1]=='1': self.aggregator = lambda x: torch.sum(x, dim=1) # x shape ntask,nsample,dim
        else: self.aggregator = lambda x: self.aggregate(x, self.w_transform)
        # the style of the transformer pre-transf
        if mode[2]=='1':
            def create_atn():
                return nn.Sequential(nn.Linear(self.c, self.inp_dim),
                              LayerNorm(self.inp_dim),
                              relu())
            self.q_funcs = nn.ModuleList([nn.ModuleList([create_atn() for i in range(self.n_head)]) for j in range(self.n_round)])
            self.k_funcs = nn.ModuleList([nn.ModuleList([create_atn() for i in range(self.n_head)]) for j in range(self.n_round)])
            self.v_funcs = nn.ModuleList([nn.ModuleList([create_atn() for i in range(self.n_head)]) for j in range(self.n_round)])

        else:
            self.q_func, self.k_func, self.v_func = [
                nn.Sequential(nn.Linear(self.c, self.inp_dim),
                              LayerNorm(self.inp_dim),
                              )
                for _ in range(3)]

    def aggregate(self, keys, w_transform):
        """
        n,c -> d,c
        :param keys:
        :param w_transform:
        :return:
        """
        weights = w_transform(keys)  # ntask, n, d
        weights = self.normalization_func(weights.permute(0,2,1), dim=-1)  # ntask, d, n
        latents_ = torch.bmm(weights, keys)  # ntask,d,n x ntask,n,c -> ntask,d,c
        return latents_.squeeze(1) if latents_.shape[1]==1 else latents_

    def transformer(self, latents, k_func, q_func, v_func):
        keys, queries, values = k_func(latents), q_func(latents), v_func(latents)  # ntask, d, dim
        aff = torch.bmm(queries, keys.permute(0, 2, 1)) / sqrt(self.inp_dim)  # ntask,d,d
        aff = F.softmax(aff, dim=-1)
        if self.selfloop:
            ident = torch.eye(aff.shape[1]).to(device=aff.device).unsqueeze(0)
            latents = torch.bmm(aff+ident, values)  # ntask, d, dim
        else:
            latents = torch.bmm(aff, values)
        return latents

    def v2l2(self, keys, w_transform):
        """
        n,c -> d,c
        :param keys:
        :param w_transform:
        :return:
        """
        latents_ = w_transform(keys.permute(0,2,1)) # c,n > c,d
        return latents_.permute(0,2,1).squeeze(1)

    def propagate(self, inp, dump=False):
        inp = F.relu(inp) # added cuz i feel like do more non-lin before attention
        ##
        ## 1. transformer+aggregate
        ### 1.1 nxc > nxc (sum) 1xc
        ### 1.2 nxc > nxc (agg) 1xc: c,1
        ## 2. agg+transformer+agg
        ### 1.1 nxc > dxc > dxc > (sum) 1xc: c,d; c,1
        ### 1.2 nxc > dxc > dxc > (agg) 1xc: c,d; c,q
        ret = dict()
        if self.mode[0]=='0': # only nxc with direct agg
            n_nodes = inp
            out = self.aggregator(n_nodes)
            out = self.dim_red(out)
        elif self.mode[0]=='1':
            n_nodes = self.transformer(inp, self.k_func, self.q_func, self.v_func)
            out = self.aggregator(n_nodes)
            out = self.dim_red(out)
        elif self.mode[0]=='2': #x2
            n_nodes = self.aggregate(inp, self.w2_transform)

            # latents = self.transformer(n_nodes, self.k_func, self.q_func, self.v_func)
            # out = self.aggregator(latents)

            in_latents = n_nodes
            for j in range(self.n_round):
                heads = list()
                for i in range(self.n_head):
                    out_latents = self.transformer(in_latents, self.k_funcs[j][i], self.q_funcs[j][i], self.v_funcs[j][i])
                    heads.append(out_latents)
                if self.n_head>1:
                    heads = torch.cat(heads, dim=-1) # ntask, d, nhead*dim
                    out_latents = self.w3_transforms[j].forward(heads)
                else:
                    out_latents = heads[0]
                    # out_latents = self.gcn(out_latents, self.k_funcs[j][-1], self.q_funcs[j][-1], self.v_funcs[j][-1])
                in_latents = out_latents

            out = self.aggregator(out_latents)

            out = self.dim_red(out)

        elif self.mode[0]=='3':
            n_nodes = self.aggregate(inp, self.w2_transform)
            n_nodes = self.normalization_func(n_nodes, dim=-1)  # normalize over the c channel
            latents = self.transformer(n_nodes, self.k_func, self.q_func, self.v_func)
            out = self.aggregator(latents)
            out = self.dim_red(out)

        elif self.mode[0]=='4':
            latents_ = self.aggregate(inp, self.w2_transform)
            # latents = self.normalization_func(latents_, dim=-1)  # normalize over the c channel
            latents = latents_
            keys, queries, values = self.k_func(latents), self.q_func(latents), self.v_func(latents)  # ntask, d, dim
            aff = torch.bmm(queries, keys.permute(0, 2, 1)) / sqrt(self.inp_dim)  # ntask,d,d
            aff = F.softmax(aff, dim=-1)
            latents = values + torch.bmm(aff, values)  # ntask, d, dim
            if self.direct_transform:
                ret = self.v2l2(latents, self.w_transform)
            else:
                ret = self.aggregate(latents, self.w2_transform)
            out = self.dim_red(ret)
        ret['out'] = out
        return ret

class RecurrentEncoder(FlattenMlp):
    '''
    encode context via recurrent network
    '''

    def __init__(self,
                 *args,
                 **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = in_.size()
        out = in_.view(task * seq, feat)

        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            out = self.hidden_activation(out)

        out = out.view(task, seq, -1)
        out, (hn, cn) = self.lstm(out, (self.hidden, torch.zeros(self.hidden.size()).to(ptu.device)))
        self.hidden = hn
        # take the last hidden state to predict z
        out = out[:, -1, :]

        # output layer
        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def reset(self, num_tasks=1):
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)




