import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu


def _product_of_gaussians(mus, sigmas_squared):
    """
    compute mu, sigma of product of gaussians
    :param mus:
    :param sigmas_squared:
    :return: tuple of 1d scalars
    """
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class PEARLAgent(nn.Module):

    def __init__(self,
                 latent_dim,
                 nets,
                 discount=0.99,
                 policy_mean_reg_weight=1e-3,
                 policy_std_reg_weight=1e-3,
                 policy_pre_activation_weight=0.,
                 soft_target_tau=1e-2,
                 **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.context_encoder, self.policy, self.qf1, self.qf2, self.vf = nets
        self.target_vf = self.vf.copy()
        self.networks = nets + [self.target_vf]

        self.recurrent = kwargs['recurrent']
        self.use_ib = kwargs['use_information_bottleneck']
        self.use_graphenc = kwargs['use_graphencoder']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.discount = discount

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.prior_learn = kwargs['learn_prior']
        if self.prior_learn:
            prior_means = torch.zeros((kwargs['num_tasks'], latent_dim)).cuda(ptu.device)
            prior_vars = torch.ones((kwargs['num_tasks'], latent_dim)).cuda(ptu.device)
            self.prior_means = nn.Parameter(prior_means)
            self.prior_vars = nn.Parameter(prior_vars)

        self.clear_z()

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        self.context_encoder.reset(num_tasks)

    def trans_z(self, other):
        self.z_means = other.z_means
        self.z_vars = other.z_vars
        self.sample_z()

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def update_context(self, inputs, use_no):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])
        if use_no:
            data = torch.cat([o, a, r, no], dim=-1)
        else: data = torch.cat([o,a,r], dim=-1)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1) # at dim of step

    def compute_kl_div(self, indices=None):
        ''' compute KL( q(z|c) || r(z) ) '''
        if indices is None:
            prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
            posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
            kl_div_sum = torch.sum(torch.stack(kl_divs))
        else:
            priors = [torch.distributions.Normal(mu, var) for mu,var in zip(torch.unbind(self.prior_means[indices]), torch.unbind(self.prior_vars[indices]))]
            posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in
                          zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post,prior in zip(posteriors, priors)]
            kl_div_sum = torch.sum(torch.stack(kl_divs))

        return kl_div_sum

    def infer_posterior(self, context, require_mid=False, dump=False):
        """
        compute q(z|c) as a function of input context and sample new z from it
        self.z_means/z_vars/z modified 2d ntask,latent_dim
        :param context:
        :return: None
        """
        params = self.context_encoder.forward(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size) # ntask,nstep,ndim
        mid = params
        ret = None
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            if self.use_graphenc:
                ret = self.context_encoder.propagate(params, dump=dump)
                params = ret['out']
                self.z_means = params[..., :self.latent_dim]
                self.z_vars = torch.clamp(F.softplus(params[..., self.latent_dim:]), min=1e-7)
            else:
                mu = params[..., :self.latent_dim]
                sigma_squared = F.softplus(params[..., self.latent_dim:]) # ntask,nstep,ndim
                # list of ntask tuples, each 1d scalar
                z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
                self.z_means = torch.stack([p[0] for p in z_params]) # 2d, ntask,latent_dim
                self.z_vars = torch.stack([p[1] for p in z_params]) # 2d, ntask,latent_dim
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=-1)
        self.sample_z()
        if require_mid: return mid
        if dump: return ret
        return None

    def sample_z(self, determ=False):
        if self.use_ib and not determ:
            # list of ntask normals
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors] # list of ntask samples
            self.z = torch.stack(z) # ntask, latent_dim
        else:
            self.z = self.z_means

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs, z], dim=-1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z)
        q2 = self.qf2(obs, actions, task_z)
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _take_step(self, indices, batch_data, context, rew_scale=1., alpha=1., ndetached=dict(), extra_exp_data=None):

        num_tasks = len(indices)
        is_explorer = context is None
        # data is (task, batch, feat)
        if extra_exp_data is not None:
            obs, actions, rewards, next_obs, terms = [torch.cat((x,y),dim=1) for x,y in zip(batch_data, extra_exp_data)]
        else: obs, actions, rewards, next_obs, terms = batch_data
        split = batch_data[0].size(1)
        # run inference in networks
        policy_outputs, task_z, mid = self.forward(obs, context, require_mid=not is_explorer) # encode to obtain z, infer new actions as max-q
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension > 2d tensors
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        tz, dtz = task_z, task_z.detach()
        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, tz if ndetached.get('q') is not None else dtz)
        q2_pred = self.qf2(obs, actions, tz if ndetached.get('q') is not None else dtz)
        v_pred = self.vf(obs, tz if ndetached.get('v') else dtz)
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        rewards_flat = rew_scale * rewards.view(b * t, -1)
        # scale rewards for Bellman update
        terms_flat = terms.view(b * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean(
            (q2_pred - q_target) ** 2)  # use off-policy data from buffer
        # KL constraint on z if probabilistic
        if is_explorer:
            kl_div = None
        elif self.prior_learn:
            kl_div = self.compute_kl_div(indices=indices)  # div of prior and z
        else:
            kl_div = self.compute_kl_div()

        # V-learning: on-policy data from the online inferred actions
        # compute min Q on the new actions
        if split is not None: # the split is used because policy is trained only on the rl buffer
            obs, new_actions, tz, dtz, log_pi, policy_mean, policy_log_std, v_pred = \
                [x.view(t,b,-1)[:,:split] for x in [obs, new_actions, tz, dtz, log_pi, policy_mean, policy_log_std, v_pred]]
        min_q_new_actions = self._min_q(obs, new_actions, tz if ndetached.get('a') else dtz) # used in actor loss, assumed detached in pearl
        log_pi = log_pi*alpha
        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())

        policy_loss = (
                -v_target
        ).mean() # max q-ent

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        # pre_tanh_value = policy_outputs[-1]
        # pre_activation_reg_loss = self.policy_pre_activation_weight * (
        #     (pre_tanh_value ** 2).sum(dim=1).mean()
        # )
        policy_reg_loss = mean_reg_loss + std_reg_loss# + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        return kl_div, qf_loss, vf_loss, policy_loss, log_pi, mid

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(self, obs, context, require_mid=False):
        ''' given context, get statistics under the current policy of a set of observations '''
        mid = None
        if context is not None:
            mid = self.infer_posterior(context, require_mid=require_mid)
        self.sample_z()
        task_z = self.z # ntask, dim

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        # task_z = [z.repeat(b, 1) for z in task_z]
        task_z = task_z.unsqueeze(1).repeat(1,b,1).view(t*b,-1)

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=-1)
        policy_outputs = self.policy(in_, reparameterize=True, return_log_prob=True)

        return policy_outputs, task_z, mid

    def log_diagnostics(self, eval_statistics, prefix=None):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

    # @property
    # def networks(self):
    #     return [self.context_encoder, self.policy]




