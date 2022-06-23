import numpy as np

from rlkit.samplers.util import rollout
from rlkit.torch.sac.policies import MakeDeterministic


class InPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_path_length):
        self.env = env
        self.policy = policy

        self.max_path_length = max_path_length

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_online_samples(self, deterministic=False, max_samples=np.inf, max_trajs=np.inf, accum_context=True, resample=1,
                              sub_update_rate=np.inf, policy=None, need_sparse_in_context=False, use_no=False):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        :return list of dicts, int
        """
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        assert (sub_update_rate<np.inf and accum_context) or (sub_update_rate==np.inf), \
            "sub_update_rate is only enabled when accum_context allowed"
        policy = self.policy if policy is None else policy
        policy = MakeDeterministic(policy) if deterministic else policy
        paths = []
        n_steps_total = 0
        n_trajs = 0
        while n_steps_total < max_samples and n_trajs < max_trajs:
            # interact to obtain one single traj - dict of 2d arr and lists.
            path = rollout(
                self.env, policy, max_path_length=self.max_path_length, accum_context=accum_context,
                sub_update_rate=sub_update_rate, need_sparse=need_sparse_in_context, use_no=use_no)
            # save the latent context that generated this trajectory
            ctx = None if not hasattr(policy, 'z') else  policy.z.detach().cpu().numpy()
            path['context'] = ctx
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1
            # don't we also want the option to resample z ever transition?
            if n_trajs % resample == 0 and  hasattr(policy, 'sample_z'):
                policy.sample_z(determ=deterministic) # resample from the same posterior
        return paths, n_steps_total

