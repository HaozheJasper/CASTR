import numpy as np
from gym import spaces
from gym import Env
import scipy.stats as ss

from . import register_env


@register_env('point-robot')
class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self, randomize_tasks=False, n_tasks=2, hollow=False, **kwargs):

        if randomize_tasks:
            np.random.seed(1337)
            goals = [[np.random.uniform(-1., 1.), np.random.uniform(-1., 1.)] for _ in range(n_tasks)]
        elif hollow:
            np.random.seed(1337)
            goals0 = [[np.random.uniform(-1., .9), np.random.uniform(-1., 1.)] for _ in range(80)]
            goals1 = [[np.random.uniform(.9, 1.), np.random.uniform(-1., 1.)] for _ in range(20)]
            goals = goals0 + goals1
        else:
            # some hand-coded goals for debugging
            locs, scales = [8.5, 5.],[0.5,0.5]
            xys = np.array([(1,1),(-1,-1),(1,-1),(-1,1)]) # 4,2
            locses = [xys*l for l in locs] # list of 4,2
            locses.append(np.zeros((1,2)))
            locs = np.concatenate(locses, axis=0) # 9,2
            scales = np.array(scales).reshape((1,2)).repeat((locs.shape[0],1)) # 9,2
            n_components = locs.shape[0]
            weights = np.ones(n_components, dtype=float) / float(n_components)
            mixture_indices = np.random.choice(n_components, size=n_tasks, replace=True, p=weights)
            samples = [ss.norm.rvs(loc=locs[i], scale=scales[i]) for i in mixture_indices]
            samples = np.array(samples).clip(-10,10)
            goals = [g / 10. for g in samples]
        self.goals = goals

        self.reset_task(0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal = self.goals[idx]
        self.reset()

    def get_all_task_idx(self):
        return range(len(self.goals))

    def reset_model(self):
        # reset to a random location on the unit square
        self._state = np.random.uniform(-1., 1., size=(2,))
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action): # action: 2, ob: 2
        self._state = self._state + action
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)


@register_env('sparse-point-robot')
class SparsePointEnv(PointEnv):
    '''
     - tasks sampled from unit half-circle
     - reward is L2 distance given only within goal radius

     NOTE that `step()` returns the dense reward because this is used during meta-training
     the algorithm should call `sparsify_rewards()` to get the sparse rewards
     '''
    def __init__(self, randomize_tasks=False, n_tasks=2, goal_radius=0.2, **kwargs):
        super().__init__(randomize_tasks, n_tasks)
        self.goal_radius = goal_radius

        if randomize_tasks:
            np.random.seed(1337)
            radius = 1.0
            angles = np.linspace(0, np.pi, num=n_tasks)
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            goals = np.stack([xs, ys], axis=1)
            np.random.shuffle(goals)
            goals = goals.tolist()

        self.goals = goals
        self.reset_task(0)

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r

    def reset_model(self):
        self._state = np.array([0, 0])
        return self._get_obs()

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            sparse_reward += 1
        d.update({'sparse_reward': sparse_reward})
        return ob, reward, done, d
