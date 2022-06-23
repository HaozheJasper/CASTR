import time
from collections import OrderedDict

import gtimer as gt
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter
from os import path as osp
import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger, eval_util
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.samplers.util import rollout
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.agent import PEARLAgent
from torch.nn import functional as F


# from rlkit.core.rl_algorithm import MetaRLAlgorithm


class PEARLSoftActorCritic:
    def __init__(
            self,
            #### MetaRLAlgo
            env,
            train_tasks, # list of indices
            eval_tasks, # list of indices
            nets,
            latent_dim,
            regularizers=None,
            meta_batch=64,
            num_iterations=100,
            num_train_steps_per_itr=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_consec=100,
            num_evals=10,
            num_steps_per_eval=1000,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=1000,
            replay_buffer_size=1000000,
            reward_scale=1,
            num_exp_traj_eval=1,
            update_post_train=1,
            eval_deterministic=True,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            render_eval_paths=False,
            dump_eval_paths=False,
            plotter=None,
            #########
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            clf_lr=1e-3,
            rewfn_lr=1e-3,
            transfn_lr=1e-3,
            discr_lr=1e-3,
            prior_lr=1e-3,
            kl_lambda=1.,
            a_ndetached=None,
            e_ndetached=None,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            sparse_rewards=False,
            exp_name=None,
            rand_policy=None,
            **kwargs

    ):

        ########## MetaRLAlgo
        self.env = env
        self.agent = nets[0]
        self.explorer = nets[1]
        # z2t,saz2r,saz2s regularizes z
        # s2t, saz2r, saz2s serves as extra rewards
        possible_regularizers = ['z2t', 'saz2r', 'saz2s']
        self.regularizers = regularizers
        self.clf, self.rew_fn, self.trans_fn = [regularizers.get(k) for k in possible_regularizers]
        tmp_nets = [x for x in regularizers.values() if x is not None]
        self.networks = self.agent.networks + self.explorer.networks[1:]+tmp_nets # context, policy, qf1, qf2, vf, tvf *2
        # self.exploration_agent = self.agent  # Can potentially use a different policy purely for exploration rather than also solving tasks, currently not being used
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        # self.num_steps_posterior = num_steps_posterior
        self.num_steps_consec = num_steps_consec
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        # self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.update_post_train = update_post_train
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment
        self.use_nobs = kwargs['no']
        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.plotter = plotter
        self.act_sampler = InPlacePathSampler(
            env=env,
            policy=self.agent,
            max_path_length=self.max_path_length,
        )
        # assume that there must be an explorer
        ## TODO incorporate none explorer scenario
        assert self.explorer is not None
        self.exp_sampler = InPlacePathSampler(
                env=env,
                policy=self.explorer,
                max_path_length=self.max_path_length,
            )

        # separate replay buffers for
        # - training RL update
        # - training encoder update
        self.replay_buffer = MultiTaskReplayBuffer(
            self.replay_buffer_size,
            env,
            self.train_tasks,
        ) # essentially 2d table for each task

        self.enc_replay_buffer = MultiTaskReplayBuffer(
            self.replay_buffer_size,
            env,
            self.train_tasks,
        )

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []
        self.task_idx = None # only assigned to real values during and after training
        #########
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.pretrain_n_epoch = 5
        self.rand_policy = rand_policy

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.rew_criterion = nn.MSELoss()
        self.trans_criterion = nn.MSELoss()
        self.sfid_criterion = nn.CrossEntropyLoss()
        self.clf_criterion = nn.CrossEntropyLoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards

        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.agent.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.agent.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.agent.vf.parameters(),
            lr=vf_lr,
        )
        self.policy2_optimizer = optimizer_class(
            self.explorer.policy.parameters(),
            lr=policy_lr,
        )
        self.qf12_optimizer = optimizer_class(
            self.explorer.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf22_optimizer = optimizer_class(
            self.explorer.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf2_optimizer = optimizer_class(
            self.explorer.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.agent.context_encoder.parameters(),
            lr=context_lr,
        )

        if self.agent.use_graphenc:
            sigma_squared_bound = torch.ones((), device=ptu.device)*self.embedding_mini_batch_size
            self.sigma_squared_bound = nn.Parameter(sigma_squared_bound)
            self.bound_optimizer = optimizer_class([self.sigma_squared_bound], lr=prior_lr)
        if self.clf is not None: self.clf_optimizer = optimizer_class(self.clf.parameters(), lr=clf_lr)
        if self.rew_fn is not None: self.rew_optimizer = optimizer_class(self.rew_fn.parameters(), lr=rewfn_lr)
        if self.trans_fn is not None: self.trans_optimizer = optimizer_class(self.trans_fn.parameters(), lr=transfn_lr)
        # if self.sfidelity is not None: self.sfid_optimizer = optimizer_class(self.sfidelity.parameters(), lr=discr_lr)

        self.prior_learn = kwargs['learn_prior']
        if self.prior_learn : self.prior_optimizer = optimizer_class([self.agent.prior_means, self.agent.prior_vars], lr=prior_lr)
        if a_ndetached is None: # by default
            self.act_ndetached = dict(q=True) # v,a is false
        else: self.act_ndetached = a_ndetached
        if e_ndetached is None:
            self.exp_ndetached = dict() # neither is true
        else: self.exp_ndetached = e_ndetached

        self.itr_cnt = 0
        logdir = osp.join('tb',exp_name)
        logger.log(logdir)
        self.writer = SummaryWriter(logdir)
        if hasattr(env, 'tasks'):
            logger.log('\n'.join(str(x) for x in env.tasks))
        else:
            logger.log('\n'.join(str(x) for x in env.goals))
        self.cmd_params = kwargs
        self.max_returns = list()
    ###### MetaRLAlgo
    def make_exploration_policy(self, policy):
        return policy

    def make_eval_policy(self, policy):
        return policy

    def sample_task(self, is_eval=False):
        """
        sample task randomly
        """
        if is_eval:
            idx = np.random.randint(len(self.eval_tasks))
        else:
            idx = np.random.randint(len(self.train_tasks))
        return idx

    def train(self):
        """
        meta-training loop
        """
        self.pretrain()
        params = self.get_epoch_snapshot(-1) # dict of params
        logger.save_itr_params(-1, params) # save params for each net
        gt.reset()
        gt.set_def_unique(False) # not unique time tamps
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ): # the training loop
            # self.ratio = (it_+1)/self.num_iterations
            self.ratio = -(1-it_/self.num_iterations)**3+1 # poly decay
            self._start_epoch(it_) # prep for new epoch
            self.training_mode(True)
            if it_ == 0:
                logger.log('collecting initial pool of data for train and eval')
                logger.log(f'initial steps = {self.num_initial_steps} x {len(self.train_tasks)}')
                # temp for evaluating
                for idx in self.train_tasks:
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.prior_collect_data(self.num_initial_steps) # only exp one traj
            # Sample data from train tasks.
            logger.log(f'collect data = {self.num_steps_prior+self.num_steps_consec} x {self.num_tasks_sample}')
            for i in range(self.num_tasks_sample): # sample tasks and update their data
                idx = np.random.randint(len(self.train_tasks)) # assume train indices are consecutive
                self.task_idx = idx
                self.env.reset_task(idx)
                self.enc_replay_buffer.task_buffers[idx].clear()

                # 1exp and 3exp enabled
                if self.num_steps_prior > 0: # prior once and posterior once with offpolicy data for encoder
                    # # use only prior samples, because the encoder buffer is not supposed to have data
                    self.prior_collect_data(self.num_steps_prior)  # only exp one traj
                if self.num_steps_consec > 0: # prior once and posterior several with onpolicy data for encoder
                    for _ in range(self.num_steps_prior//self.max_path_length):
                        self.post_collect_data(self.num_steps_consec, 1, self.update_post_train, onpolicy=self.cmd_params['onpolicy'])

            # Sample train tasks and compute gradient updates on parameters.
            logger.log(f'train for {self.num_train_steps_per_itr} steps')
            for train_step in range(self.num_train_steps_per_itr):
                indices = np.random.choice(self.train_tasks, self.meta_batch)
                self._do_training(indices)
                self._n_train_steps_total += 1
            gt.stamp('train')

            self.training_mode(False)

            # eval
            self._try_to_eval(it_)
            gt.stamp('eval')

            self._end_epoch()

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        # tenv = self.env._wrapped_env
        # bounds = [[ptu.from_numpy(x.low),ptu.from_numpy(x.high)] for x in [ tenv.action_space]]
        # def foo(inp):
        #     ls = list()
        #     for ori,b in zip(inp,bounds):
        #         lo,hi = b
        #         tmp = (ori-lo)/(hi-lo)
        #         ls.append(tmp)
        #     return ls[0]
        ###### fidelity does not make sense
        # if self.sfidelity is not None:
        #     gt.reset()
        #     gt.set_def_unique(False)  # not unique time tamps
        #     for epoch in range(self.pretrain_n_epoch):
        #         print(f'pretrain_epoch_{epoch}')
        #         for task in self.train_tasks:
        #             self.task_idx = task
        #             self.env.reset_task(task)
        #             self.prior_collect_data(self.num_steps_consec, act=False, exp=True, policy=self.rand_policy) # want to keep this as initial data for encoder
        #         print(f'collected {self.num_steps_prior} x {len(self.train_tasks)}')
        #         for step in range(200):
        #             indices = np.random.choice(self.train_tasks, size=len(self.train_tasks))
        #             ##### num of samples for each task
        #             bs, self.embedding_batch_size = self.embedding_batch_size,32
        #             obs, actions, rewards, next_obs, terms = self.sample_data(indices, encoder=True)
        #             # actions = foo([actions])
        #             self.embedding_batch_size = bs
        #             #######
        #             gt_idx_ = torch.from_numpy(indices).long().to(ptu.device)
        #             gt_idx = gt_idx_.unsqueeze(1).repeat(1, obs.size(1)).view(-1)  # ntask, nstep
        #             gt_encoding = F.one_hot(gt_idx, len(self.train_tasks)).float()
        #             inp = torch.cat((obs, actions, next_obs, rewards), dim=-1)  # ntaskxnstep, dim
        #             inp = inp.view(-1, inp.size(-1))
        #             fid_loss = 1e2*self.regularization(inp, self.sfidelity, gt_idx, self.sfid_criterion)
        #
        #             self.writer.add_scalar('fidelity_fn_pre', fid_loss, self.itr_cnt)
        #             self.itr_cnt += 1
        #             self.sfid_optimizer.zero_grad()
        #             fid_loss.backward()
        #             self.sfid_optimizer.step()
        #         print(fid_loss)
        #         indices = np.random.choice(self.train_tasks, size=80)
        #         ##### num of samples for each task
        #         bs, self.embedding_batch_size = self.embedding_batch_size, 32
        #         obs, actions, rewards, next_obs, terms = self.sample_data(indices, encoder=True)
        #         # actions = foo([actions])
        #         self.embedding_batch_size = bs
        #         #######
        #         gt_idx_ = torch.from_numpy(indices).long().to(ptu.device)
        #         gt_idx = gt_idx_.unsqueeze(1).repeat(1, obs.size(1)).view(-1)  # ntask, nstep
        #         # gt_encoding = F.one_hot(gt_idx, len(self.train_tasks)).float()
        #         inp = torch.cat((obs, actions, next_obs, rewards), dim=-1)  # ntaskxnstep, dim
        #         inp = inp.view(-1, inp.size(-1))
        #         prediction = self.sfidelity(inp).argmax(dim=-1)
        #         print((prediction==gt_idx).sum(), inp.shape[0])
        # self.itr_cnt = 0

    def collect_task_posterior(self, save_dir, mixture=False):
        """
        collect the z posterior for each tasks
        """
        prefix = 'mix' if mixture else ''
        gt.reset()
        gt.set_def_unique(False)  # not unique time tamps
        self._current_path_builder = PathBuilder()
        self.training_mode(False)
        if hasattr(self.env, 'tasks'):
            np.save(osp.join(save_dir, 'tasks.pth'), self.env.tasks)
        else:
            np.save(osp.join(save_dir, 'tasks.pth'), self.env.goals)
        # if it_ == 0:
        logger.log('collecting initial pool of data for train and eval')
        logger.log(f'steps for all task = (prior:{self.num_steps_prior}+poster:{self.num_steps_consec}) x {len(self.train_tasks)}')
        # temp for evaluating
        for idx in self.train_tasks:
            logger.log(f'on task {idx}')
            self.task_idx = idx
            self.env.reset_task(idx)
            # collect with prior, both exp and act requires some rollouts
            if self.num_steps_prior > 0:  # prior once and posterior once with offpolicy data for encoder
                # # use only prior samples, because the encoder buffer is not supposed to have data
                for _ in range(self.num_steps_prior // self.max_path_length):
                    # 1 prior 1 post for both, 2 traj for both buffer
                    self.prior_collect_data()  # only exp one traj
            if self.num_steps_consec > 0:  # prior once and posterior several with onpolicy data for encoder
                self.post_collect_data(self.num_steps_consec, 1, self.update_post_train,
                                       onpolicy=self.cmd_params['onpolicy'])

        obs, act, rew, _, _ = self.sample_data(self.train_tasks, encoder=True)
        enc_inp = self.prepare_encoder_data(obs, act, rew)
        self.agent.infer_posterior(enc_inp)
        posteriors = self.agent.z_means, self.agent.z_vars # ntask, dim
        names = [osp.join(save_dir,prefix+n) for n in ['trn_means.pth','trn_vars.pth']]
        for p,n in zip(posteriors, names): torch.save(p.cpu(), n)

        logger.log('clear the buffer for eval tasks')
        for task in self.train_tasks:
            self.enc_replay_buffer.clear_buffer(task)
            self.replay_buffer.clear_buffer(task)

        for idx in range(len(self.eval_tasks)):
            logger.log(f'on eval task {idx}')
            self.task_idx = idx
            self.env.reset_task(idx)
            # collect with prior, both exp and act requires some rollouts
            if self.num_steps_prior > 0:  # prior once and posterior once with offpolicy data for encoder
                # # use only prior samples, because the encoder buffer is not supposed to have data
                for _ in range(self.num_steps_prior // self.max_path_length):
                    # 1 prior 1 post for both, 2 traj for both buffer
                    self.prior_collect_data()  # only exp one traj
            if self.num_steps_consec > 0:  # prior once and posterior several with onpolicy data for encoder
                self.post_collect_data(self.num_steps_consec, 1, self.update_post_train,
                                       onpolicy=self.cmd_params['onpolicy'])

        obs, act, rew, _, _ = self.sample_data(np.array(self.eval_tasks)-self.eval_tasks[0], encoder=True)
        enc_inp = self.prepare_encoder_data(obs, act, rew)
        self.agent.infer_posterior(enc_inp)
        posteriors = self.agent.z_means, self.agent.z_vars # ntask, dim
        names = [osp.join(save_dir,prefix+n) for n in ['val_means.pth','val_vars.pth']]
        for p,n in zip(posteriors, names): torch.save(p.cpu(), n)

    # def collect_data(self, num_exp_samples, resample_z_rate, update_posterior_rate, onpolicy=False, accum_context=False, sub_update_rate=np.inf):
    #     """
    #     get trajectories from current env in batch mode with given policy
    #     collect complete trajectories until the number of collected transitions >= num_samples
    #
    #     :param agent: policy to rollout
    #     :param num_exp_samples: total number of transitions to sample
    #     :param resample_z_rate: how often to resample latent context z (in units of trajectories)
    #     :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
    #     :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
    #
    #     to enable sub update of posterior in a single path, enable sub_update_rate and accum_context
    #     """
    #     # start from the prior
    #     self.agent.clear_z() # would clear the context
    #     self.explorer.clear_z()
    #     num_transitions = 0
    #     # prior samples into rl buffer
    #     n_samples = 0
    #     if update_posterior_rate==np.inf: # only collect prior samples when update posterior is not allowed
    #         paths, n_samples = self.act_sampler.obtain_online_samples(max_trajs=1,
    #                                                                   accum_context=False,
    #                                                                   resample=resample_z_rate)  # list of dicts of 2d arrs, int
    #         self.replay_buffer.add_paths(self.task_idx, paths)
    #     num_transitions += n_samples
    #     cnt_transitions = 0
    #     while cnt_transitions < num_exp_samples:
    #         # obtain one or more trajs, this integral batch does not include posterior update
    #         paths, n_samples = self.exp_sampler.obtain_online_samples(max_samples=num_exp_samples - cnt_transitions,
    #                                                                   max_trajs=update_posterior_rate,
    #                                                                   accum_context=accum_context,
    #                                                                   sub_update_rate=sub_update_rate,
    #                                                                   resample=resample_z_rate) # list of dicts of 2d arrs, int
    #         cnt_transitions += n_samples
    #         self.enc_replay_buffer.add_paths(self.task_idx, paths) # prior and post samples into encoder buffer
    #         # print(self.task_idx, self.enc_replay_buffer.task_buffers[self.task_idx]._size)
    #         if update_posterior_rate != np.inf :
    #             if onpolicy:
    #                 # 2d arrays are made 3d
    #                 paths = ptu.np_to_pytorch_batch(paths[0])
    #                 obs = paths['observations'][None, ...]
    #                 act = paths['actions'][None, ...]
    #                 rewards = paths['rewards'][None, ...]
    #                 nobs = paths['next_observations'][None, ...]
    #                 context = self.prepare_encoder_data(obs, act, rewards, nobs)
    #             else:
    #                 context = self.prepare_context(self.task_idx) # use uncorrelated data, and z is not autoregressive
    #             self.explorer.infer_posterior(context) # and the explorer set its z
    #             self.agent.trans_z(self.explorer)
    #             # post samples into rl buffer
    #             paths, n_samples = self.act_sampler.obtain_online_samples(max_trajs=1,
    #                                                                       accum_context=False,
    #                                                                       resample=resample_z_rate)  # list of dicts of 2d arrs, int
    #             self.replay_buffer.add_paths(self.task_idx, paths)
    #             num_transitions += n_samples
    #
    #     # num transitions for rl samples, cnt transitions for encoder buffer
    #     num_transitions += cnt_transitions
    #     self._n_env_steps_total += num_transitions
    #
    #     gt.stamp('sample')

    def prior_collect_data(self, num_samples, act=True, exp=True, policy=None):
        """
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_exp_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer

        to enable sub update of posterior in a single path, enable sub_update_rate and accum_context
        """
        # start from the prior
        # print('prior',num_samples)
        self.agent.clear_z() # would clear the context
        self.explorer.clear_z()
        num_transitions = 0
        # prior samples into rl buffer
        n_samples = 0
        if act:
            paths, n_samples = self.act_sampler.obtain_online_samples(#max_trajs=1,
                                                                      max_samples=num_samples,
                                                                      accum_context=False,
                                                                      policy=policy,
                                                                      )  # list of dicts of 2d arrs, int
            self.replay_buffer.add_paths(self.task_idx, paths)
            num_transitions += n_samples
        if exp:
            # obtain one or more trajs, this integral batch does not include posterior update
            paths, n_samples = self.exp_sampler.obtain_online_samples(#max_trajs=1,
                                                                      max_samples=num_samples,
                                                                      accum_context=False,
                                                                      policy=policy,
                                                                      )  # list of dicts of 2d arrs, int
            # num_transitions += n_samples
            self.enc_replay_buffer.add_paths(self.task_idx, paths)  # prior and post samples into encoder buffer
            # print(self.task_idx, self.enc_replay_buffer.task_buffers[self.task_idx]._size)
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')

    def post_collect_data(self, num_exp_samples, resample_z_rate, update_posterior_rate, onpolicy=False, sub_update_rate=np.inf):
        """
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_exp_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer

        to enable sub update of posterior in a single path, enable sub_update_rate and accum_context
        """
        # start from the prior
        # print('post', num_exp_samples)
        assert  update_posterior_rate!= np.inf
        self.agent.clear_z() # would clear the context
        self.explorer.clear_z()
        num_transitions = 0
        # prior samples into rl buffer
        n_samples = 0
        cnt_transitions = 0
        while cnt_transitions < num_exp_samples:
            # obtain one or more trajs, this integral batch does not include posterior update
            left = num_exp_samples - cnt_transitions
            onepath = update_posterior_rate*self.max_path_length
            samples_this_iter = min(left, onepath)
            paths, n_samples = self.exp_sampler.obtain_online_samples(max_samples=samples_this_iter,#num_exp_samples - cnt_transitions,
                                                                      # max_trajs=update_posterior_rate,
                                                                      accum_context=True,
                                                                      use_no=self.use_nobs,
                                                                      sub_update_rate=sub_update_rate,
                                                                      resample=resample_z_rate,
                                                                      need_sparse_in_context=self.sparse_rewards) # list of dicts of 2d arrs, int
            if cnt_transitions!=0: # do not add prior samples
                self.enc_replay_buffer.add_paths(self.task_idx, paths)  # prior and post samples into encoder buffer
                # print(self.task_idx, self.enc_replay_buffer.task_buffers[self.task_idx]._size)
            cnt_transitions += n_samples

            if update_posterior_rate != np.inf :
                if onpolicy:
                    # clen = len(self.explorer.context)
                    # indices = np.random.choice(np.arange(clen), min(clen, self.embedding_mini_batch_size), replace=False)
                    context = self.explorer.context
                else:
                    context = self.prepare_context(self.task_idx) # use uncorrelated data, and z is not autoregressive
                self.explorer.infer_posterior(context) # and the explorer set its z
                self.agent.trans_z(self.explorer)
                # post samples into rl buffer
                paths, n_samples = self.act_sampler.obtain_online_samples(max_samples=1*self.max_path_length,
                                                                          accum_context=False,
                                                                          resample=resample_z_rate)  # list of dicts of 2d arrs, int
                self.replay_buffer.add_paths(self.task_idx, paths)
                num_transitions += n_samples

        # num transitions for rl samples, cnt transitions for encoder buffer
        # num_transitions += cnt_transitions
        self._n_env_steps_total += num_transitions

        gt.stamp('sample')

    def _try_to_eval(self, epoch):
        # save like rendered data
        logger.save_extra_data(self.get_extra_data_to_save(epoch)) # ?? no training_env
        if self._can_evaluate():
            trn_ret, tst_ret = self.evaluate(epoch)
            current_max = np.array(self.max_returns)
            comp = current_max-tst_ret<0
            update_param = np.any(comp)
            if len(self.max_returns)<3: self.max_returns.append(tst_ret)
            elif update_param:
                print(comp)
                replace_where, = np.where(comp)
                ind = np.arange(3)[replace_where][0]
                self.max_returns[ind] = tst_ret
            self.max_returns = sorted(self.max_returns)
            if update_param:
                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation, )

    def _start_epoch(self, epoch):
        """
        preparation for each new epoch
        :param epoch:
        :return:
        """
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    ##### Snapshotting utils #####

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def _check_test_eff(self, indices, num_steps):
        task_ret_growths = list()
        for idx in indices:
            print(f'on task {idx}')
            paths = self.incremental_paths(idx, num_steps, exp_determ=True)  # alist of path
            all_ret = [eval_util.get_average_returns([p]) for p in paths]
            task_ret_growths.append(all_ret)
        return np.mean(task_ret_growths, axis=0)  # ntask, nstep_exp's resulting return

    def collect_paths(self, idx, epoch, exp_determ=False, sub_update_rate=np.inf):
        """
        explore with num_exp traj, update z with the remaining trajs (incrementally), return all paths
        only used in evaluations
        :param idx:
        :param epoch:
        :param actor_num_run:
        :return: list of dicts
        """
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z() # would clear the context
        self.explorer.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0
        for i in range(self.num_steps_per_eval//self.max_path_length-1): # deterministic only means the policy not the encoder
            path, num = self.exp_sampler.obtain_online_samples(deterministic=exp_determ,  # pearl use self.eval_determ
                                                               # max_trajs=1,
                                                               max_samples=1*self.max_path_length,
                                                               accum_context=True,
                                                               sub_update_rate=sub_update_rate,
                                                               need_sparse_in_context=self.sparse_rewards,
                                                               use_no=self.use_nobs)
            if self.dump_eval_paths:
                logger.save_extra_data(path, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, 0))
            self.explorer.infer_posterior(self.explorer.context)  # becomes incremental since context is accum
            # if i>=self.num_exp_traj_eval:
            #     self.explorer.infer_posterior(self.explorer.context) # becomes incremental since context is accum
        self.agent.trans_z(self.explorer)
        for _ in range(2):
            path, num = self.act_sampler.obtain_online_samples(deterministic=self.eval_deterministic,
                                                               # max_trajs=1,
                                                               max_samples=1*self.max_path_length,
                                                               resample=1, # resample from the same post for each run
                                                               accum_context=False, sub_update_rate=np.inf)
            paths += path
        # num_transitions += num
        # num_trajs += 1

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal  # goal

        # save the paths for visualization, only useful for point mass


        return paths
    #
    def incremental_paths(self, idx, num_steps, exp_determ=False, sub_update_rate=np.inf):
        """
        explore with num_exp traj, update z with the remaining trajs (incrementally), return all paths
        only used in evaluations
        :param idx:
        :param epoch:
        :param actor_num_run:
        :return: list of dicts
        """
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z()  # would clear the context
        self.explorer.clear_z()
        # using the prior
        path, num = self.act_sampler.obtain_online_samples(deterministic=self.eval_deterministic,
                                                           max_trajs=1,
                                                           resample=1,
                                                           # resample from the same post for each run
                                                           accum_context=False, sub_update_rate=np.inf,
                                                           use_no=self.use_nobs)
        paths = path
        num_transitions = 0
        num_trajs = 0
        goal = self.env._goal
        epaths = list()
        rets = list()
        for i in range(
                num_steps // self.max_path_length ):  # deterministic only means the policy not the encoder
            path, num = self.exp_sampler.obtain_online_samples(deterministic=exp_determ,  # pearl use self.eval_determ
                                                               max_samples=1*self.max_path_length,
                                                               # sparse=self.sparse_rewards,
                                                               accum_context=True, sub_update_rate=sub_update_rate,
                                                               use_no=self.use_nobs)
            # TODO randomly select from the contexts
            # if i >= self.num_exp_traj_eval:
            ret = self.explorer.infer_posterior(
                self.explorer.context, dump=self.dump_eval_paths)  # becomes incremental since context is accum

            path[0]['goal'] = goal  # goal
            if self.dump_eval_paths:
                epaths.append(path)
                rets.append(ret)
            self.agent.trans_z(self.explorer)
            path, num = self.act_sampler.obtain_online_samples(deterministic=self.eval_deterministic,
                                                               max_trajs=1,
                                                               resample=1,
                                                               # resample from the same post for each run
                                                               accum_context=False, sub_update_rate=np.inf,
                                                               use_no=self.use_nobs)
            paths.extend(path)
            num_transitions += num
            num_trajs += 1

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(epaths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, 0, 0))
            logger.save_extra_data(rets, path='eval_trajectories/task{}-gnn'.format(idx))

        return paths

    def _do_eval(self, indices, epoch, exp_included=False):
        """

        :param indices:
        :param epoch:
        :return: final returns - a list of ntask average final return;
        online returns - a list of ntask lists of x trajs return
        """
        final_returns = []
        online_returns = []
        for idx in indices:
            all_rets = [] # a list of several evals, each of which is [the average of ] a list of the returns for each path in one eval
            for r in range(self.num_evals): # eval once (explore once), obtain 2 traj
                paths = self.collect_paths(idx, epoch, exp_determ=self.cmd_params['exp_determ']) # get a list of dicts
                tmp = [eval_util.get_average_returns([p]) for p in paths]
                # print(tmp)
                all_rets.append(np.mean(tmp)) # append a list of total returns of all paths
            # average return of final traj.
            # print(all_rets)
            final_returns.append(np.mean(all_rets)) # take the average of, a list of the total return of the final path of all trials
            if exp_included:
                # record online returns for the first n trajectories
                n = min([len(a) for a in all_rets]) # the length of the shortest exp-test rollouts
                all_rets = [a[:n] for a in all_rets] # a list of lists of returns, align the length of all trials
                all_rets = np.mean(np.stack(all_rets), axis=0)  # avg return across the evaluations (each n rollouts), avg return per nth rollout
                online_returns.append(all_rets)
        if exp_included:
            n = min([len(t) for t in online_returns])
            online_returns = [t[:n] for t in online_returns] # a list of ntask lists of x trajs return
        else: online_returns = None
        return final_returns, online_returns

    def evaluate(self, epoch):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### sample trajectories from prior for debugging / visualization
        if self.dump_eval_paths:
            # 100 arbitrarily chosen for visualizations of point_robot trajectories
            # just want stochasticity of z, not the policy
            self.agent.clear_z() # set z to samples of prior, clear the context
            prior_paths, _ = self.act_sampler.obtain_online_samples(deterministic=self.eval_deterministic,
                                                                    max_samples=self.max_path_length * 20,
                                                                    accum_context=False,
                                                                    resample=1,
                                                                    use_no=self.use_nobs)
            logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        ### train tasks
        # eval on a subset of **train tasks** for speed
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
        ### eval train tasks with posterior sampled from the training replay buffer
        train_returns = []
        paths = None
        for idx in indices:
            self.task_idx = idx
            self.env.reset_task(idx)
            paths = []
            # evaluate performance with once-updated z posterior several times : off-policy data
            for _ in range(self.num_steps_per_eval // self.max_path_length):
                context = self.prepare_context(idx)
                self.agent.infer_posterior(context) # set z to samples of posterior
                p, _ = self.act_sampler.obtain_online_samples(deterministic=self.eval_deterministic,
                                                              max_samples=self.max_path_length,
                                                              accum_context=False,
                                                              max_trajs=1,
                                                              resample=np.inf,
                                                              use_no=self.use_nobs)
                paths += p

            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards

            train_returns.append(eval_util.get_average_returns(paths)) # append the average total return among several trials of the current task
        train_returns = np.mean(train_returns) # the average return across multiple tasks
        self.writer.add_scalar('offp_ret', train_returns, global_step=epoch)
        ### eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_online_returns = self._do_eval(indices, epoch) # use the subset of train tasks
        # eval_util.dprint('train online returns')
        # eval_util.dprint(train_online_returns)

        ### test tasks
        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch)
        # eval_util.dprint('test online returns')
        # eval_util.dprint(test_online_returns)

        # save the final posterior
        self.agent.log_diagnostics(self.eval_statistics)

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(paths, prefix=None) # only evaluate the last list of paths?

        avg_train_return = np.mean(train_final_returns) # average return of final traj with on-policy data of train tasks
        avg_test_return = np.mean(test_final_returns)
        # avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        # avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        self.writer.add_scalar('onp_trn', avg_train_return, global_step=epoch)
        self.writer.add_scalar('onp_val', avg_test_return, global_step=epoch)
        # self.writer.add_histogram('onp_trn_all', avg_train_online_return, global_step=epoch)
        # self.writer.add_histogram('onp_val_all', avg_test_online_return, global_step=epoch)
        self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
        # logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        # logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()
        return avg_train_return, avg_test_return

    ###### Torch stuff #####
    # @property
    # def networks(self):
    #     return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        """
        device is set in ptu, and net vars are moved to the device
        :param device: not needed if ptu.set_gpu_mode() previously
        :return: None
        """
        if device is None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    def prepare_encoder_data(self, *inp):
        """
        prepare context for encoding
        :param obs: 3d
        :param act: 3d
        :param rewards: 3d
        :return: 3d
        """
        # for now we embed only observations and rewards
        # assume obs and rewards are (task, batch, feat)
        task_data = torch.cat(inp, dim=2)
        return task_data

    def prepare_context(self, idx):
        """
        sample context from replay buffer and prepare it
        :param idx:
        :return: a 3d tensor
        """
        # in the beginning the encoder buffer only receives one traj
        size = self.enc_replay_buffer.task_buffers[idx]._size
        assert size>=self.max_path_length, f'current size {size}<{self.max_path_length}'
        size = min(size, self.embedding_batch_size)
        batch = ptu.np_to_pytorch_batch(
            self.enc_replay_buffer.random_batch(idx, batch_size=size, sequence=False)) # TODO: sequence arg should change
        # 2d arrays are made 3d
        obs = batch['observations'][None, ...]
        act = batch['actions'][None, ...]
        nobs = batch['next_observations'][None, ...]
        # any context should use sparse rewards mode buffer data if necessary
        rewards = batch['sparse_rewards'][None, ...] if self.sparse_rewards else batch['rewards'][None, ...]
        if self.use_nobs:
            context = self.prepare_encoder_data(obs, act, rewards, nobs)
        else:
            context = self.prepare_encoder_data(obs, act, rewards)
        return context

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size
        # data for encoder should use sparse ones if necessary (assured by need_sparse
        batch = self.sample_data(indices, encoder=True, exp=False) # a list of 3d tensors

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))
        self.explorer.clear_z(num_tasks=len(indices))

        consumed = 0
        random_input_num = False
        for i in range(num_updates):
            if random_input_num:
                if i<num_updates-1:
                    rsize = np.random.randint(mb_size//2, mb_size*2)
                    start, end = consumed, consumed+rsize
                    consumed += rsize
                    mini_batch = [x[:, start:end] for x in batch]
                else:
                    mini_batch = [x[:, consumed:] for x in batch]
            else:
                mini_batch = [x[:, i * mb_size: i * mb_size + mb_size, :] for x in batch] # take mini batch of each of the tensors

            self._take_step(indices, mini_batch)

            # stop backprop
            self.agent.detach_z()

    ##### Data handling #####
    def sample_data(self, indices, encoder=False, exp=False, curiosity=None, z=None, raw_fac=1., cur_fac=1e-3, ent_fac=1.,
                    fid_fac=1e2, trans_fac=1e2, rewf_fac=1e2, raw_scale=None, context=None):
        """

        :param indices:
        :param encoder: sample data from encoder
        :param exp: the data is used for explorer
        :param ent:
        :param curiosity:
        :param include:
        :param z:
        :param raw_scale: to align with q-err
        :param raw_fac: factor for raw rewards
        :param cur_fac:
        :param ent_fac:
        :return:
        """
        # collect data from multiple tasks for the meta-batch
        # decide which reward should we use:
        # 1. sparse reward: only for encoder in sparse reward setting
        # 2. entropies: when ent is enabled, does not assume any other things
        # 3. reward: only when it is for encoder, but entr is enabled exclusively
        # assert: when exp is on, curiosity must not be None
        if curiosity is not None:
            keys = ['use_sparse','use_raw', 'use_entropy', 'use_qerr', 'use_fidelity', 'use_trans', 'use_rewfn']
            use_sparse, use_raw, need_entr, use_qerr, use_fid, use_trans, use_rewf = [curiosity.get(k) for k in keys]
            include = sum(curiosity.values()) > 1  # more than 2 fields are used
            curiosity = True
        else:
            need_entr = False

        need_sparse = self.sparse_rewards and (encoder and (not exp or use_sparse ))
        # when do we not need raw reward: when it is not for encoder, always need raw;
        # when it is for encoder but not for exp, spare may replace raw
        # when it is for exp, when not use raw and not use qerr
        need_not_raw = encoder and ((not exp and need_sparse) or (exp and not use_raw and not use_qerr and not use_fid and not use_rewf))
        # not only when use_raw, but also some cur requires it
        obs, actions, rewards, next_obs, terms, entropies, sparses = [], [], [], [], [], [], []
        # if curiosity!=False:
        #     cur_baseline = curiosity
        #     curiosity = True
        for idx in indices:
            if encoder:
                batch = ptu.np_to_pytorch_batch(
                    self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size,
                                                        sequence=self.recurrent))
            else:
                batch = ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size))
            o = batch['observations'][None, ...]
            a = batch['actions'][None, ...]

            if need_sparse: sparses.append(batch['sparse_rewards'][None, ...])

            if not need_not_raw: rewards.append(batch['rewards'][None, ...])

            if need_entr: entropies.append(batch['entropies'][None, ...])

            no = batch['next_observations'][None, ...]
            t = batch['terminals'][None, ...]

            obs.append(o)
            actions.append(a)
            next_obs.append(no)
            terms.append(t)
        obs = torch.cat(obs, dim=0)
        actions = torch.cat(actions, dim=0)
        next_obs = torch.cat(next_obs, dim=0)
        terms = torch.cat(terms, dim=0)
        if need_sparse: sparses = torch.cat(sparses, dim=0)
        if not need_not_raw: raw = torch.cat(rewards, dim=0)
        if need_entr: entropies = ent_fac*torch.cat(entropies, dim=0)
        if not encoder:
            rewards = raw # sparse is only used for encoder
        else: # encoder
            if not exp: # for the encoder
                if need_sparse: rewards = sparses
                else: rewards = raw
            else: # for explorer, they definitely have curisoity
                with torch.no_grad():
                    if use_qerr:
                        z_ = z.unsqueeze(1).repeat(1, obs.size(1), 1)
                        curr= cur_fac * torch.abs(
                                    raw*raw_scale + self.agent.discount * (1. - terms) * self.agent.target_vf.forward(next_obs,
                                                                                                                z_) -
                                    self.agent.qf1.forward(obs, actions, z_))

                    if use_fid: # sasr->pred_t scores, fetch the current task score
                        if self.use_nobs:
                            fid_inp = torch.cat((obs, actions, next_obs, raw), dim=-1)
                        else:
                            fid_inp = torch.cat((obs, actions, raw), dim=-1)
                        tmp = self.sfidelity(fid_inp, context)
                        # gt_idx = ptu.from_numpy(indices).long().view(-1,1).repeat(1,obs.size(1)).unsqueeze(2)
                        # tmp = torch.gather(predicted_scores, dim=-1, index=gt_idx)
                        fidr = fid_fac * tmp
                        # print('fidr', tmp.mean(), tmp.max(), tmp.min(), tmp.shape)
                        if use_qerr:
                            # curr *= (1.-self.ratio)
                            # fidr *= self.ratio
                            fidr *= 1 # for walker, cuz the scale for qerr is 1e4, and fidr is mult by 1e2
                        # print(self.ratio, fidr.mean(), fidr.max(), fidr.min())
                    if use_trans: # saz->pred_ns, use the err
                        z_ = z.unsqueeze(1).repeat(1, obs.size(1), 1)
                        transr = trans_fac*torch.abs(self.trans_fn(torch.cat((obs, actions, z_), dim=-1))-next_obs).mean(dim=-1, keepdim=True)
                    if use_rewf: # saz->pred_r, use the err
                        z_ = z.unsqueeze(1).repeat(1, obs.size(1), 1)
                        rewfr = rewf_fac * torch.abs(
                            self.rew_fn(torch.cat((obs, actions, z_), dim=-1)) - raw) # use_raw is enough
                if not need_not_raw: raw = raw_fac*raw
                if need_sparse:
                    sparses = raw_fac*sparses

                if (self.itr_cnt+1)%100==0: # do not place this after calc rewards, some references may change
                    if use_qerr: self.writer.add_histogram('qerr', curr, self.itr_cnt)
                    # if need_entr: self.writer.add_histogram('entropies', entropies, self.itr_cnt)
                    # if use_trans: self.writer.add_histogram('trans_rew', transr, self.itr_cnt)
                    # if use_rewf: self.writer.add_histogram('rewf_rew', rewfr, self.itr_cnt)
                    # if use_fid: self.writer.add_histogram('fid_rew', fidr, self.itr_cnt)
                    if not need_not_raw: self.writer.add_histogram('raw_rew', raw, self.itr_cnt)
                    if need_sparse:
                        self.writer.add_histogram('sparse_rew', sparses, self.itr_cnt)

                if not include: # exclusive
                    # TODO: can check to see if sparse rewards can be set as rewards to guide the explorer
                    if need_entr: rewards = entropies
                    elif use_qerr: rewards = curr
                    elif use_rewf: rewards = rewfr
                    elif use_trans: rewards = transr
                    elif use_fid: rewards = fidr
                    elif use_sparse and need_sparse: rewards = sparses
                    elif not need_not_raw: rewards = raw
                else:
                    init_shape = [obs.size(0), obs.size(1),1]
                    rewards = torch.zeros(init_shape).to(ptu.device)
                    if need_sparse and use_sparse: rewards += sparses # why is this sparses bounded to rewards
                    if use_raw: rewards += raw
                    if need_entr: rewards += entropies
                    if use_qerr: rewards += curr
                    if use_trans: rewards += transr
                    if use_fid: rewards += fidr
                    if use_rewf: rewards += rewfr

        return [obs, actions, rewards, next_obs, terms]  # list of 3d tensors, ntask,nstep,dim

    def regularization(self, inp, fn, targets, criterion, return_outputs=False):
        """
        :return:
        """
        out = fn(inp)
        if return_outputs:
            return out, criterion(out, targets)
        else:
            return criterion(out, targets)

    def read_curiosity_code(self, code):
        """

        :param code:
        :return: a dict of true or false
        """
        # rparams: e: entropy; q:qerr; r:rewfn; t:transfn; f:fidelity sar-z; s:sparse; b:basic rewards
        fileds2name = dict(e='use_entropy',q='use_qerr',b='use_raw',s='use_sparse', f='use_fidelity', t='use_trans', r='use_rewfn')

        if '+' in code: # if + concats fields
            fields = code.split('+')
            options = {fileds2name[k]:True for k in fields}
        else: # assert there is single char
            options = {fileds2name[code]:True}

        all_keys = set(fileds2name.values())
        for k in all_keys-set(options.keys()): # some keys is not present in options
            options[k] = False
        return options

    def sfidelity(self, samples, context):
        """

        :param samples: ntask, nstep, dim
        :param context:
        :return:
        """
        samples = self.agent.context_encoder.forward(samples)
        similarities = list()
        for sam, con in zip(torch.unbind(samples, dim=0), torch.unbind(context, dim=0)):
            # sam,con = [F.normalize(x,dim=-1) for x in [sam,con]]
            neg_sim = -torch.matmul(sam, con.transpose(1,0)) # the cos dist, \in [-1,1]
            v,idx = neg_sim.max(dim=-1)
            similarities.append(v)
        return torch.stack(similarities, dim=0).unsqueeze(-1)

    def _take_step(self, indices, mini_batch):
        obs_enc, act_enc, rewards_enc, nobs_enc, _ = mini_batch  # 3d tensors
        # print(obs_enc.shape, act_enc.shape)
        if self.use_nobs:
            context = self.prepare_encoder_data(obs_enc, act_enc, rewards_enc, nobs_enc)
        else: context = self.prepare_encoder_data(obs_enc, act_enc, rewards_enc)
        agt_data = self.sample_data(indices, encoder=False)
        # extra_exp_data = self.sample_data(indices, encoder=True, exp=False)

        actor_scale = self.reward_scale
        exp_scale = 1. # achieve by separately designating cur_fac and ent_fac?
        kl_div_agt, qf_loss_agt, vf_loss_agt, agt_loss, _, mid = self.agent._take_step(indices, agt_data, context, rew_scale=actor_scale,
                                                                                    ndetached=self.act_ndetached, extra_exp_data=None)
        # TODO sample different context for explorer
        cur_code, raw_fac, cur_fac, fid_fac= [self.cmd_params['exp_rew'].get(k) for k in ['code','raw_fac','cur_fac','fid_fac']]
        if raw_fac is None: raw_fac = self.reward_scale
        # if ent_fac is None: ent_fac = self.reward_scale
        # if is_qerr: is_qerr = torch.sqrt(qf_loss_agt/2.).mean()
        # curiosity: entropy; qerr; rew_fn; trans_fn; s_fid; raw
        self.explorer.trans_z(self.agent)
        exp_data = self.sample_data(indices, encoder=True, exp=True, curiosity=self.read_curiosity_code(cur_code),
                                    raw_fac=raw_fac, cur_fac=cur_fac, fid_fac=fid_fac, raw_scale=raw_fac, z=self.explorer.z.detach(),
                                    context=mid)
        _, qf_loss_exp, vf_loss_exp, exp_loss, logp, _ = self.explorer._take_step(indices, exp_data, None, rew_scale=exp_scale,
                                                                               ndetached=self.exp_ndetached)

        # regularizers: rew-mse; trans-mse; clf:xent; sfidelity:mse
        gt_idx_ = torch.from_numpy(indices).long().to(ptu.device)  # ntask,: expect the tensor inherit the data type
        obs, act, rew, nobs = agt_data[:4]
        ztmp = self.agent.z.unsqueeze(1).repeat(1, obs.size(1), 1)  # ntask,nstep,dim
        # reward function
        regularization_loss = torch.zeros(()).to(ptu.device)
        # if self.agent.use_graphenc:
        #     sigma = F.softplus(self.agent.z_vars).mean(dim=-1) # ntask,
        #     nsamples = obs_enc.size(1)
        #     tmp = sigma*nsamples-self.sigma_squared_bound
        #     scale_penalty = F.relu(tmp).mean()
        #     # hope that sigma abs could be smaller than k/nsamples
        #     regularization_loss += scale_penalty # filter out all that smaller than zero and average the positive ones
        #     self.writer.add_scalar('scale_penalty', scale_penalty, self.itr_cnt)
        if self.rew_fn is not None:
            inp = torch.cat((obs, act, ztmp), dim=-1) # need ztmp's grad
            rew_loss = 1e2*self.regularization(inp, self.rew_fn, rew, self.rew_criterion) # 1e-1
            self.writer.add_scalar('rew_fn', rew_loss, self.itr_cnt)
            regularization_loss += rew_loss
        if self.trans_fn is not None:
            inp = torch.cat((obs, act, ztmp), dim=-1)
            trans_loss = 1e3*self.regularization(inp, self.trans_fn, nobs, self.trans_criterion) # 1e-2
            self.writer.add_scalar('trans_fn', trans_loss, self.itr_cnt)
            regularization_loss += trans_loss
        if self.clf is not None:
            z_samples = list()
            for _ in range(4):
                self.agent.sample_z()
                z_samples.append(self.agent.z)
            z_samples = torch.cat(z_samples, dim=0)  # ntaskxnstep, dim
            gt_idx_clf = gt_idx_.repeat(4)  # ntaskx4
            task_preds, clf_loss = self.regularization(z_samples, self.clf, gt_idx_clf, self.clf_criterion, # 4
                                                       return_outputs=True)
            self.writer.add_scalar('clf_fn', clf_loss, self.itr_cnt)
            regularization_loss += clf_loss
        # found that if we regress from s to t, z is not regularized here
        self.context_optimizer.zero_grad()
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        if self.rew_fn is not None: self.rew_optimizer.zero_grad()
        if self.trans_fn is not None: self.trans_optimizer.zero_grad()
        if self.clf is not None: self.clf_optimizer.zero_grad()
        if self.prior_learn: self.prior_optimizer.zero_grad()
        if self.agent.use_graphenc: self.bound_optimizer.zero_grad()


        kl_loss = self.kl_lambda * kl_div_agt
        # print(kl_loss, self.kl_lambda)
        (kl_loss+qf_loss_agt+regularization_loss).backward()


        if self.agent.use_graphenc: self.bound_optimizer.step()
        if self.prior_learn: self.prior_optimizer.step()
        if self.clf is not None: self.clf_optimizer.step()
        if self.trans_fn is not None: self.trans_optimizer.step()
        if self.rew_fn is not None: self.rew_optimizer.step()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        if (self.itr_cnt + 1) % 3000 == 0:
            try:
                if exp_scale!=1.: self.writer.add_histogram(f'exp_rew_bef_{exp_scale}', exp_data[2], self.itr_cnt)
                self.writer.add_histogram('logp', logp, self.itr_cnt)
            except Exception as e:
                repr(e)
        # if (self.itr_cnt +1) % 300 == 0:
        #     try:
        #         self.writer.add_scalar('qf_agt', qf_loss_agt, self.itr_cnt)
        #         self.writer.add_scalar('qf_exp', qf_loss_exp, self.itr_cnt)
        #         self.writer.add_scalar('vf_agt', vf_loss_agt, self.itr_cnt)
        #         self.writer.add_scalar('vf_exp', vf_loss_exp, self.itr_cnt)
        #         self.writer.add_scalar('agt', agt_loss, self.itr_cnt)
        #         self.writer.add_scalar('exp', exp_loss, self.itr_cnt)
        #         self.writer.add_scalar('kl', kl_loss, self.itr_cnt)
        #     except Exception as e:
        #         repr(e)
        self.itr_cnt += 1

        self.vf_optimizer.zero_grad()
        vf_loss_agt.backward()
        self.vf_optimizer.step()
        self.agent._update_target_network()

        self.policy_optimizer.zero_grad()
        agt_loss.backward()
        self.policy_optimizer.step()

        self.qf12_optimizer.zero_grad()
        self.qf22_optimizer.zero_grad()
        qf_loss_exp.backward()
        self.qf12_optimizer.step()
        self.qf22_optimizer.step()

        self.vf2_optimizer.zero_grad()
        vf_loss_exp.backward()
        self.vf2_optimizer.step()
        self.explorer._update_target_network()

        self.policy2_optimizer.zero_grad()
        exp_loss.backward()
        self.policy2_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div_agt)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss_agt))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss_agt))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                agt_loss
            ))

    def _e_step(self, indices, mini_batch):
        obs_enc, act_enc, rewards_enc, nobs_enc, _ = mini_batch  # 3d tensors
        # print(obs_enc.shape, act_enc.shape)
        if self.use_nobs:
            context = self.prepare_encoder_data(obs_enc, act_enc, rewards_enc, nobs_enc)
        else:
            context = self.prepare_encoder_data(obs_enc, act_enc, rewards_enc)
        agt_data = self.sample_data(indices, encoder=False)

        actor_scale = self.reward_scale
        exp_scale = 1.  # achieve by separately designating cur_fac and ent_fac?
        kl_div_agt, qf_loss_agt, vf_loss_agt, agt_loss, _, mid = self.agent._take_step(indices, agt_data, context,
                                                                                       rew_scale=actor_scale,
                                                                                       ndetached=self.act_ndetached,
                                                                                       extra_exp_data=None,
                                                                                       true_qz=True and self.em_soft_targetnet)
        # TODO sample different context for explorer
        cur_code, raw_fac, cur_fac, fid_fac = [self.cmd_params['exp_rew'].get(k) for k in
                                               ['code', 'raw_fac', 'cur_fac', 'fid_fac']]
        if raw_fac is None: raw_fac = self.reward_scale

        # regularizers: rew-mse; trans-mse; clf:xent; sfidelity:mse
        gt_idx_ = torch.from_numpy(indices).long().to(ptu.device)  # ntask,: expect the tensor inherit the data type
        obs, act, rew, nobs = agt_data[:4]
        ztmp = self.agent.z.unsqueeze(1).repeat(1, obs.size(1), 1)  # ntask,nstep,dim
        # reward function
        regularization_loss = torch.zeros(()).to(ptu.device)
        if self.agent.use_graphenc:
            sigma = F.softplus(self.agent.z_vars).mean(dim=-1)  # ntask,
            nsamples = obs_enc.size(1)
            tmp = sigma * nsamples - self.sigma_squared_bound
            scale_penalty = F.relu(tmp).mean()
            # hope that sigma abs could be smaller than k/nsamples
            regularization_loss += scale_penalty  # filter out all that smaller than zero and average the positive ones
            self.writer.add_scalar('scale_penalty', scale_penalty, self.itr_cnt)
        if self.rew_fn is not None:
            inp = torch.cat((obs, act, ztmp), dim=-1)  # need ztmp's grad
            rew_loss = 1e2 * self.regularization(inp, self.rew_fn, rew, self.rew_criterion)  # 1e-1
            self.writer.add_scalar('rew_fn', rew_loss, self.itr_cnt)
            regularization_loss += rew_loss
        if self.trans_fn is not None:
            inp = torch.cat((obs, act, ztmp), dim=-1)
            trans_loss = 1e3 * self.regularization(inp, self.trans_fn, nobs, self.trans_criterion)  # 1e-2
            self.writer.add_scalar('trans_fn', trans_loss, self.itr_cnt)
            regularization_loss += trans_loss
        if self.clf is not None:
            z_samples = list()
            for _ in range(4):
                self.agent.sample_z()
                z_samples.append(self.agent.z)
            z_samples = torch.cat(z_samples, dim=0)  # ntaskxnstep, dim
            gt_idx_clf = gt_idx_.repeat(4)  # ntaskx4
            task_preds, clf_loss = self.regularization(z_samples, self.clf, gt_idx_clf, self.clf_criterion,  # 4
                                                       return_outputs=True)
            self.writer.add_scalar('clf_fn', clf_loss, self.itr_cnt)
            regularization_loss += clf_loss
        # found that if we regress from s to t, z is not regularized here
        self.context_optimizer.zero_grad()

        if self.rew_fn is not None: self.rew_optimizer.zero_grad()
        if self.trans_fn is not None: self.trans_optimizer.zero_grad()
        if self.clf is not None: self.clf_optimizer.zero_grad()
        if self.prior_learn: self.prior_optimizer.zero_grad()
        if self.agent.use_graphenc: self.bound_optimizer.zero_grad()

        kl_loss = self.kl_lambda * kl_div_agt
        if self.post_kl:
            # print('post kl')
            ############### online context
            bdata = self.sample_data(indices, encoder=True, sequence=True)
            online_context = self.prepare_encoder_data(*(bdata[:4] if self.use_nobs else bdata[:3]))
            online_post_mean, online_post_var = self.agent.get_posterior(online_context,
                                                                         true_qz=True and self.em_soft_targetnet)
            online_kl_div = self.agent.kl_div(online_post_mean, online_post_var, self.agent.z_means, self.agent.z_vars)
            self.writer.add_scalar('online_kl', online_kl_div, self.itr_cnt)
            kl_loss += 5e-2 * online_kl_div # use agent.z because agent and explorer share z and agent's z is sure to be attached.
        # print(kl_loss, self.kl_lambda)
        (kl_loss + qf_loss_agt + regularization_loss).backward()

        if self.agent.use_graphenc: self.bound_optimizer.step()
        if self.prior_learn: self.prior_optimizer.step()
        if self.clf is not None: self.clf_optimizer.step()
        if self.trans_fn is not None: self.trans_optimizer.step()
        if self.rew_fn is not None: self.rew_optimizer.step()

        self.context_optimizer.step()

        self.writer.add_scalar('kl', kl_loss, self.itr_cnt)
        self.itr_cnt += 1

    def _m_step(self, indices, mini_batch):
        obs_enc, act_enc, rewards_enc, nobs_enc, _ = mini_batch  # 3d tensors
        # print(obs_enc.shape, act_enc.shape)
        if self.use_nobs:
            context = self.prepare_encoder_data(obs_enc, act_enc, rewards_enc, nobs_enc)
        else: context = self.prepare_encoder_data(obs_enc, act_enc, rewards_enc)
        agt_data = self.sample_data(indices, encoder=False)
        # extra_exp_data = self.sample_data(indices, encoder=True, exp=False)

        actor_scale = self.reward_scale
        exp_scale = 1. # achieve by separately designating cur_fac and ent_fac?
        kl_div_agt, qf_loss_agt, vf_loss_agt, agt_loss, _, mid = self.agent._take_step(indices, agt_data, context, rew_scale=actor_scale,
                                                                                    ndetached=self.act_ndetached, extra_exp_data=None)
        # TODO sample different context for explorer
        cur_code, raw_fac, cur_fac, fid_fac= [self.cmd_params['exp_rew'].get(k) for k in ['code','raw_fac','cur_fac','fid_fac']]
        if raw_fac is None: raw_fac = self.reward_scale
        # if ent_fac is None: ent_fac = self.reward_scale
        # if is_qerr: is_qerr = torch.sqrt(qf_loss_agt/2.).mean()
        # curiosity: entropy; qerr; rew_fn; trans_fn; s_fid; raw
        self.explorer.trans_z(self.agent)
        exp_data = self.sample_data(indices, encoder=True, exp=True, curiosity=self.read_curiosity_code(cur_code),
                                    raw_fac=raw_fac, cur_fac=cur_fac, fid_fac=fid_fac, raw_scale=raw_fac, z=self.explorer.z.detach(),
                                    context=mid)
        _, qf_loss_exp, vf_loss_exp, exp_loss, logp, _ = self.explorer._take_step(indices, exp_data, None, rew_scale=exp_scale,
                                                                               ndetached=self.exp_ndetached)

        # found that if we regress from s to t, z is not regularized here
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        qf_loss_agt.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        self.vf_optimizer.zero_grad()
        vf_loss_agt.backward()
        self.vf_optimizer.step()
        self.agent._update_target_network()

        self.policy_optimizer.zero_grad()
        agt_loss.backward()
        self.policy_optimizer.step()

        self.qf12_optimizer.zero_grad()
        self.qf22_optimizer.zero_grad()
        qf_loss_exp.backward()
        self.qf12_optimizer.step()
        self.qf22_optimizer.step()

        self.vf2_optimizer.zero_grad()
        vf_loss_exp.backward()
        self.vf2_optimizer.step()
        self.explorer._update_target_network()

        self.policy2_optimizer.zero_grad()
        exp_loss.backward()
        self.policy2_optimizer.step()

        if (self.itr_cnt + 1) % 3000 == 0:
            self.writer.add_histogram(f'exp_rew_bef_{exp_scale}', exp_data[2], self.itr_cnt)
            self.writer.add_histogram('logp', logp, self.itr_cnt)
        # if (self.itr_cnt +1) % 300 == 0:
        #     self.writer.add_scalar('qf_agt', qf_loss_agt, self.itr_cnt)
        #     self.writer.add_scalar('qf_exp', qf_loss_exp, self.itr_cnt)
        #     self.writer.add_scalar('vf_agt', vf_loss_agt, self.itr_cnt)
        #     self.writer.add_scalar('vf_exp', vf_loss_exp, self.itr_cnt)
        #     self.writer.add_scalar('agt', agt_loss, self.itr_cnt)
        #     self.writer.add_scalar('exp', exp_loss, self.itr_cnt)

        self.itr_cnt += 1

    def get_epoch_snapshot(self, epoch):
        """
        ordered dict of params, each key corresponds to one network
        :param epoch:
        :return: single ordereddict
        """
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.agent.qf1.state_dict(),
            qf2=self.agent.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.agent.vf.state_dict(),
            target_vf=self.agent.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            qf12=self.explorer.qf1.state_dict(),
            qf22=self.explorer.qf2.state_dict(),
            policy2=self.explorer.policy.state_dict(),
            vf2=self.explorer.vf.state_dict(),
            target_vf2=self.explorer.target_vf.state_dict(),
            clf=self.clf,
            # sfidelity=self.sfidelity,
            rew_fn=self.rew_fn,
            trans_fn=self.trans_fn
        )

        return snapshot
