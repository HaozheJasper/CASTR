"""
Launcher for experiments with PEARL

"""
import os
import os.path as osp
import pathlib
import numpy as np
import click
import json
import torch
from sys import platform
from rlkit.torch.sac.policies import TanhGaussianPolicy, UniformPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, GraphEncoder
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config
import pandas as pd
import subprocess
import json
from rlkit.core import logger
import torch.functional as F

def experiment(variant, debug):
    envname = variant['env_name']
    if 'rand' in envname:
        os.environ['rp-envs'] = 'y'
        if platform == 'linux':
            for key in ['LD_LIBRARY_PATH','MUJOCO_PY_MJPRO_PATH']:
                line = os.environ[key]
                os.environ[key] = line.replace('mjpro150', 'mjpro131')
        else:  # win32 can only use up to 150
            line = os.environ['MUJOCO_PY_MJPRO_PATH']
            os.environ['MUJOCO_PY_MJPRO_PATH'] = line.replace('mjpro150', 'mjpro131')
    elif 'point' in envname:
        os.environ['point'] = 'y'
    # must set the environment var before the envs module loaded
    from rlkit.envs import ENVS
    from rlkit.envs.wrappers import NormalizedBoxEnv
    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    ## **find usage get_all_task, and you can check the code
    tasks = env.get_all_task_idx() # returns Range obj
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    # instantiate networks
    latent_dim = variant['latent_size']

    reward_dim = 1
    net_size = variant['net_size']
    cnet_size = variant['cnet_size']
    recurrent = False#variant['algo_params']['recurrent']
    gnn_param = variant['algo_params']['use_graphencoder']
    gnn_nlatent = variant['algo_params']['d']
    gnn_feat = variant['algo_params']['feat']
    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'],
                     variant['util_params']['gpu_id'])  # global var ptu.device is set
    context_encoder_out = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    context_encoder_in = obs_dim * 2 + action_dim + reward_dim if variant['algo_params'][
        'no'] else obs_dim + action_dim + reward_dim
    if gnn_param is not None:
        context_encoder = GraphEncoder(
            hidden_sizes=[cnet_size, cnet_size, cnet_size],
            input_size=context_encoder_in,
            output_size=context_encoder_out,
            mid_size=32 if gnn_feat is None else gnn_feat, # feature size
            num_component=20 if gnn_nlatent is None else gnn_nlatent,
            mode=gnn_param,
            selfloop=variant['self_loop']
        )
    else:
        context_encoder = MlpEncoder(
            hidden_sizes=[cnet_size, cnet_size, cnet_size],
            input_size=context_encoder_in,
            output_size=context_encoder_out,
        )
    # for MlpEncoder\FlattenMlp checkout `class Mlp` to see the default settings.

    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy( # takes different init_w 1e-3 rather than 3e-3 as in Mlp
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )

    ## explorer
    qf12 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf22 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    rand_policy = UniformPolicy(env._wrapped_env.action_space.low, env._wrapped_env.action_space.high)
    if variant['rand']: policy2 = rand_policy
    else:
        policy2 = TanhGaussianPolicy(  # takes different init_w 1e-3 rather than 3e-3 as in Mlp
            hidden_sizes=[net_size, net_size, net_size],
            obs_dim=obs_dim + latent_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
        )

    agent = PEARLAgent(
        latent_dim,
        nets=[context_encoder, policy, qf1, qf2, vf],
        **variant['algo_params']
    )
    explorer = PEARLAgent(
        latent_dim,
        nets=[context_encoder, policy2, qf12, qf22, vf2],
        **variant['algo_params']
    )
    clf = FlattenMlp(  # z to t
        hidden_sizes=[net_size, net_size],
        input_size=latent_dim,
        output_size=variant['n_train_tasks']
    )
    # sfidelity = FlattenMlp(  # s to t
    #     hidden_sizes=[128, 256],
    #     input_size=obs_dim*2+action_dim+1, # sasr2t
    #     output_size=variant['n_train_tasks'],
    #     # hidden_activation=torch.nn.SELU(),
    #     # hidden_init=torch.nn.init.xavier_uniform_
    #     # output_activation=torch.relu
    # )
    rew_pred = FlattenMlp(  # s,a,z to r
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,  # s,a,z
        output_size=1  # to a single reward
    )  # mse
    trans1 = FlattenMlp(  # s,a,z to s
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,  # s,a,z
        output_size=obs_dim  # to a single reward
    )  # mse
    trans2 = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim + obs_dim,  # s,a,z
        output_size=variant['n_train_tasks']  # to a single reward
    )
    regularizers = dict()
    reg_code = variant['regularizers']
    if 'c' in reg_code: regularizers['z2t'] = clf
    if 't' in reg_code: regularizers['saz2s'] = trans1
    if 'r' in reg_code: regularizers['saz2r'] = rew_pred
    # if 'f' in reg_code: regularizers['sasr2t'] = sfidelity # s2t > sasr2t
    if '-' in reg_code: variant['algo_params']['kl'] = False
    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))
    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir, exp_name = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id,
                                                base_log_dir=variant['util_params']['base_log_dir'])
    ####### log to a universal file
    label = subprocess.check_output(['git', 'log', '-1', '--pretty=oneline']).strip().decode('utf-8')
    df = pd.DataFrame([[exp_name, variant['util_params']['gpu_id'], variant['memo'], label]], )
    df.to_csv('exp_logger.csv', mode='a', header=False, index=False)
    ####### log to a universal file
    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, explorer],
        regularizers=regularizers,
        latent_dim=latent_dim,
        replay_buffer_size = 10000 if debug else int(variant['buffer_size']), # 1200x100
        exp_name=exp_name,
        rand_policy=rand_policy,
        **variant['algo_params']
    )

    # TODO fix load params
    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth'), map_location='cpu'))
        # qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth'), map_location='cpu'))
        # qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth'), map_location='cpu'))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth'), map_location='cpu'))
        # vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth'), map_location='cpu'))
        # qf12.load_state_dict(torch.load(os.path.join(path, 'qf12.pth'), map_location='cpu'))
        # qf22.load_state_dict(torch.load(os.path.join(path, 'qf22.pth'), map_location='cpu'))
        policy2.load_state_dict(torch.load(os.path.join(path, 'policy2.pth'), map_location='cpu'))
        # vf2.load_state_dict(torch.load(os.path.join(path, 'vf2.pth'), map_location='cpu'))
        # TODO hacky, revisit after model refactor
        # algorithm.agent.target_vf.load_state_dict(torch.load(os.path.join(path, 'target_vf.pth'), map_location='cpu'))
        # algorithm.explorer.target_vf.load_state_dict(
        #     torch.load(os.path.join(path, 'target_vf.pth'), map_location='cpu'))
        # regularizers
        # name2handle = dict(clf=clf, rew_fn=rew_pred, trans_fn=trans1)
        # exist_regs = [n for n in name2handle.keys() if os.path.exists(os.path.join(path, f'{n}.pth'))]
        # for name in exist_regs:
        #     tmp_load = torch.load(os.path.join(path, f'{name}.pth'), map_location='cpu')
        #     if tmp_load is None: continue
        #     name2handle[name].load_state_dict(tmp_load)

    if ptu.gpu_enabled():
        algorithm.to() # move net vars to the device

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    if variant['vis']: algorithm.collect_task_posterior(variant['path_to_weights'],
                                                        mixture=not variant['env_params']['randomize_tasks'])
    elif variant['eff']>0:
        if variant['algo_params']['dump_eval_paths']:
            _ = algorithm._check_test_eff(np.random.choice(algorithm.train_tasks, 10),
                                          num_steps=variant['eff'] * algorithm.max_path_length)
        else:
            trn_tasks = algorithm._check_test_eff(algorithm.train_tasks,
                                                  num_steps=variant['eff'] * algorithm.max_path_length)
            evl_tasks = algorithm._check_test_eff(algorithm.eval_tasks,
                                                  num_steps=variant['eff'] * algorithm.max_path_length)
            trn_evl = np.stack((trn_tasks, evl_tasks), axis=0)  # 2,nstep
            logger.log(trn_evl)
            np.save(os.path.join(variant['path_to_weights'], 'test_eff_2.pth'), trn_evl)

    else:
        algorithm.train()
        trn_tasks = algorithm._check_test_eff(algorithm.train_tasks,
                                              num_steps=variant['eff'] * algorithm.max_path_length)
        evl_tasks = algorithm._check_test_eff(algorithm.eval_tasks,
                                              num_steps=variant['eff'] * algorithm.max_path_length)
        trn_evl = np.stack((trn_tasks, evl_tasks), axis=0)  # 2,nstep
        logger.log(trn_evl)
        np.save(os.path.join('output', exp_name, 'test_eff.pth'), trn_evl)

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

# python launch_experiment.py "configs\cheetah-vel.json" --rew_fac 1,1 --memo "reproduce chetv"
@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--buffer', default=100000)
@click.option('--debug', is_flag=True, default=False)
@click.option('--vis', is_flag=True, default=False) # collect the posterior params for visualization
@click.option('--eff', is_flag=False, default=0)
@click.option('--nobs', is_flag=True, default=False)
@click.option('--restore', is_flag=False) # the params restoration path
@click.option('--mixture', is_flag=True) # use a 9-mixture task sampling
@click.option('--hollow', is_flag=True)
@click.option('--gnn', is_flag=False, default="221_32,32") # 221_ should be the best
@click.option('--sloop', is_flag=False, default="True")
@click.option('--lp', is_flag=True)
@click.option('--params', is_flag=False, default='ldn_q') # cruiosity: s b e q r t f
@click.option('--rew_fac', is_flag=False, default='10,1e2') # cur,fid
@click.option('--reg', is_flag=False, default='') # regularizer: c t f r
@click.option('--ad', is_flag=False, default='') # a_ndetached, q,v,a
@click.option('--ed', is_flag=False, default='')
@click.option('--rand', is_flag=True, default=False) # random exp
@click.option('--dump', is_flag=True, default=False) # activate only when eff enabled
# @click.option('--no')
@click.option('--memo', is_flag=False, default='')
def main(config, gpu, buffer, debug, vis, eff, restore, mixture, hollow, gnn, sloop, lp, params, rew_fac, reg, ad, ed, rand, nobs, dump, memo):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu
    variant['buffer_size'] = buffer
    variant['self_loop'] = eval(sloop)
    logger.log(f'Attention: debug={debug}')
    cparams, rparams = params.split('_')
    netsize, evaldet, onp = cparams
    # rparams: e: entropy; q:qerr; r:rewfn; t:transfn; f:fidelity sar-z; s:sparse; b:basic rewards
    d = dict()
    # d['ent'] = True if 'e' in rparams else False
    # d['qerr'] = True if 'q' in rparams else False
    # d['inclusive'] = True if '+' in rparams else False
    # d['sparse'] = True if 's' in rparams else False
    d['code'] = rparams
    # factors
    facs = [eval(x) for x in rew_fac.split(',')]
    cur_fac, fid_fac = facs[-2:]
    d['cur_fac'] = cur_fac
    d['fid_fac'] = fid_fac
    d['raw_fac'] = None if len(facs) == 2 else facs[0]
    # regularizers
    variant['regularizers'] = reg
    if netsize=='l':
        variant['net_size'] = 300
        variant['cnet_size'] = 200
    if gnn=='':
        gnn, n_latent, feat =[None]*3
    else:
        gnn,param = gnn.split('_')
        n_latent,feat = [eval(x) for x in param.split(',')]

    if eff: # the eff is enabled

        # go find the variant.json
        with open(osp.join(restore, 'variant.json')) as f:
            d = json.load(f)
        gnn = d['algo_params'].get('use_graphencoder',None)
        n_latent = d['algo_params'].get('d',None)
        feat = d['algo_params'].get('feat',None)
        if gnn is not None and n_latent is None:
            tmp = gnn.split('_')
            if len(tmp)>1:
                gnn, param = tmp
                n_latent, feat = [eval(x) for x in param.split(',')]
            else:
                gnn = tmp[0]
                if len(gnn)!=3:
                    print(f'gnn={gnn}, quit')
                    exit(-1)

        variant['algo_params']['dump_eval_paths'] = dump
    #
    temp = ['q', 'v', 'a']
    a_nd, e_nd = dict(), dict()
    if ad != '':
        va = [temp[int(s)] for s in ad]
        for k in va: a_nd[k] = True
    else: a_nd = None

    if ed!='':
        ve = [temp[eval(s)] for s in ed]
        for k in ve: e_nd[k] = True
    else: e_nd = None

    variant['algo_params'].update(dict(exp_determ=evaldet=='d', onpolicy=onp=='y', exp_rew=d, use_graphencoder=gnn,
                                       d=n_latent, feat=feat,
                                       num_tasks=variant['n_train_tasks'], learn_prior=lp,
                                       a_ndetached=a_nd, e_ndetached=e_nd, no=nobs or variant['algo_params'].get('no', False)))
    variant['vis'] = vis
    variant['eff'] = eff
    variant['path_to_weights'] = restore
    variant['rand'] = rand
    variant['memo'] = memo

    if not 'point' in variant['env_name']: hollow = False
    variant['env_params']['randomize_tasks'] = not mixture and not hollow
    if 'point' in variant['env_name']: variant['env_params']['hollow'] = hollow


    if debug:
        modified = dict(
            num_initial_steps=200,
                         num_steps_prior=200,
                         num_steps_consec=200,
                        num_train_steps_per_itr=2)
        for k,v in modified.items():
            variant['algo_params'][k] = v
    experiment(variant, debug)

if __name__ == "__main__":
    main()

