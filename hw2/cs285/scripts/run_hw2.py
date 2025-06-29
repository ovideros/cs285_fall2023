import os
import time

from cs285.agents.pg_agent import PGAgent

import os
import time

import gym
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.action_noise_wrapper import ActionNoiseWrapper

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

MAX_NVIDEO = 2


def run_training_loop(cfg):
    # 初始化wandb
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.exp_name,
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes
        )
    
    logger = Logger(cfg.logdir)

    # set random seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    ptu.init_gpu(use_gpu=not cfg.ngpu, gpu_id=cfg.gpu_id)

    # make the gym environment
    env = gym.make(cfg.env_name, render_mode=None)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # add action noise, if needed
    if cfg.action_noise_std > 0:
        assert not discrete, f"Cannot use --action_noise_std for discrete environment {cfg.env_name}"
        env = ActionNoiseWrapper(env, cfg.seed, cfg.action_noise_std)

    max_ep_len = cfg.ep_len or env.spec.max_episode_steps

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if hasattr(env, "model"):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    agent = PGAgent(
        ob_dim,
        ac_dim,
        discrete,
        n_layers=cfg.l,
        layer_size=cfg.s,
        gamma=cfg.discount,
        learning_rate=cfg.lr,
        use_baseline=cfg.use_baseline,
        use_reward_to_go=cfg.rtg,
        normalize_advantages=cfg.na,
        baseline_learning_rate=cfg.blr,
        baseline_gradient_steps=cfg.bgs,
        gae_lambda=cfg.gae_lambda,
    )

    total_envsteps = 0
    start_time = time.time()

    for itr in range(cfg.n):
        print(f"\n********** Iteration {itr} ************")
        # sample `cfg.b` transitions using utils.sample_trajectories
        # make sure to use `max_ep_len`
        trajs, envsteps_this_batch = utils.sample_trajectories(env, agent.actor, cfg.b, max_ep_len)
        total_envsteps += envsteps_this_batch

        # trajs should be a list of dictionaries of NumPy arrays, where each dictionary corresponds to a trajectory.
        # this line converts this into a single dictionary of lists of NumPy arrays.
        trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}

        # train the agent using the sampled trajectories and the agent's update function
        train_info: dict = agent.update(trajs_dict['observation'], trajs_dict['action'], trajs_dict['reward'], trajs_dict['terminal'])

        if itr % cfg.scalar_log_freq == 0:
            # save eval metrics
            print("\nCollecting data for eval...")
            eval_trajs, eval_envsteps_this_batch = utils.sample_trajectories(
                env, agent.actor, cfg.eb, max_ep_len
            )

            logs = utils.compute_metrics(trajs, eval_trajs)
            # compute additional metrics
            logs.update(train_info)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs[
                    "Train_AverageReturn"
                ]

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                logger.log_scalar(value, key, itr)
                # 同时记录到wandb
                if cfg.use_wandb:
                    wandb.log({key: value}, step=itr)
            print("Done logging...\n\n")

            logger.flush()

        if cfg.video_log_freq != -1 and itr % cfg.video_log_freq == 0:
            print("\nCollecting video rollouts...")
            eval_video_trajs = utils.sample_n_trajectories(
                env, agent.actor, MAX_NVIDEO, max_ep_len, render=True
            )

            logger.log_trajs_as_videos(
                eval_video_trajs,
                itr,
                fps=fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )

    # 关闭wandb
    if cfg.use_wandb:
        wandb.finish()


@hydra.main(version_base=None, config_path="../cfg", config_name="short_names")
def main(cfg: DictConfig) -> None:
    # 映射简写到完整参数名 (如果需要的话，但为了最小化修改，我们直接在代码中使用简写)
    
    # create directory for logging
    logdir_prefix = "q2_pg_"  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = (
        logdir_prefix
        + cfg.exp_name
        + "_"
        + cfg.env_name
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    logdir = os.path.join(data_path, logdir)
    cfg.logdir = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    run_training_loop(cfg)


if __name__ == "__main__":
    if os.environ.get("ACTIVATE_CUSTOM_REPR"):
        print("--- [DEBUG] Activating custom PyTorch Tensor repr ---")
        import torch
        original_repr = torch.Tensor.__repr__
        def custom_tensor_repr(tensor):
            return f"Shape: {tensor.shape}\n{original_repr(tensor)}"
        torch.Tensor.__repr__ = custom_tensor_repr
    main()
