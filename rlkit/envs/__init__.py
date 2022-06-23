import os
import importlib


ENVS = {}


def register_env(name):
    """Registers a env by name for instantiation in rlkit."""

    def register_env_fn(fn):
        if name in ENVS:
            raise ValueError("Cannot register duplicate env {}".format(name))
        if not callable(fn):
            raise TypeError("env {} must be callable".format(name))
        ENVS[name] = fn
        return fn

    return register_env_fn


# automatically import any envs in the envs/ directory
skip_rand = os.environ.get('rp-envs') is None
point = os.environ.get('point') is not None
for file in os.listdir(os.path.dirname(__file__)):
    if skip_rand and 'rand' in file: continue  # do not include rand params env, as mjpro131 not compatible with mujoco-py 1.50
    if point and 'point' not in file: continue
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('rlkit.envs.' + module)
