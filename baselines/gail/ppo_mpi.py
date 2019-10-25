import time
import os
import numpy as np
import os.path as osp
from collections import deque

from contextlib import contextmanager

from mpi4py import MPI

import tensorflow as tf
import numpy as np

import baselines.common.tf_util as U
from baselines.common import explained_variance, zipsame, dataset, fmt_row
from baselines import logger
from baselines.common.mpi_adam import MpiAdam
from baselines.common.policies import build_policy
from baselines.common import explained_variance, set_global_seeds
# from baselines.ppo2.runner import Runner
from baselines.gail.ppo_runner import Runner

def constfn(val):
    def f(_):
        return val
    return f


def learn(*, network, env, reward_giver, expert_dataset, d_step, d_stepsize=3e-4, total_timesteps, eval_env = None, seed = None, nsteps = 2048, ent_coef = 0.0, lr = 3e-4,
          vf_coef = 0.5, max_grad_norm = 0.5, gamma = 0.99, lam = 0.95,
          log_interval = 10, nminibatches = 4, noptepochs = 4, cliprange = 0.2,
          save_interval = 0, load_path = None, model_fn = None, update_fn = None, init_fn = None, mpi_rank_weight = 1, comm = None, **network_kwargs):

    # from PPO learn
    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    policy = build_policy(env, network, **network_kwargs)

    # nenvs = env.num_envs
    nenvs = 1

    ob_space = env.observation_space
    ac_space = env.action_space

    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    if model_fn is None:
        from baselines.ppo2.model import Model
        model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                     nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                     max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight)

    if load_path is not None:
        model.load(load_path)

    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, reward_giver=reward_giver)
    if eval_env is not None:
        eval_runner = Runner(env=eval_env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    if init_fn is not None:
        init_fn()

    tfirststart = time.perf_counter()

    nupdates = total_timesteps // nbatch

    # from TRPO MPI
    nworkers = MPI.COMM_WORLD.Get_size()

    ob = model.act_model.X
    ac = model.A

    d_adam = MpiAdam(reward_giver.get_trainable_variables())

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    # from PPO
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)

        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
        if eval_env is not None:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run()

        if update % log_interval == 0 and is_mpi_root: logger.info('Done.')

        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        mblossvals = []
        logger.log("Optimizing Policy...")
        if states is None:
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else:
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.perf_counter()
        fps = int(nbatch / (tnow - tstart))

        # TRPO MPI
        logger.log("Optimizing Disciminator...")
        logger.log(fmt_row(13, reward_giver.loss_name))
        ob_expert, ac_expert = expert_dataset.get_next_batch(len(obs))
        batch_size = len(obs) // d_step
        d_losses = []
        for ob_batch, ac_batch in dataset.iterbatches((obs, actions),
                                                      include_final_partial_batch=False,
                                                      batch_size=batch_size):
            ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob_batch))
            if hasattr(reward_giver, "obs_rms"): reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
            *newlosses, g = reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
            d_adam.update(allmean(g), d_stepsize)
            d_losses.append(newlosses)

        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

        if update_fn is not None:
            update_fn(update)

        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("misc/serial_timesteps", update*nsteps)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("misc/explained_variance", float(ev))
            logger.logkv("eprewmean", safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]))
            if eval_env is not None:
                logger.logkv("eval_eprewmean", safemean([epinfo['r'] for epinfo in eval_epinfobuf]))
                logger.logkv("eval_eplenmean", safemean([epinfo['l'] for epinfo in eval_epinfobuf]))
            logger.logkv("misc/time_elapsed", tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv("loss/" + lossname, lossval)

            logger.dumpkvs()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and is_mpi_root:
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print("Saving to", savepath)
            model.save(savepath)

    return model

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)





