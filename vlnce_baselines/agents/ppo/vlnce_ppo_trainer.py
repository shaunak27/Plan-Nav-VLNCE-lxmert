# ------------------------------------------------------------------------------
# @brief: PPO trainer for the local planner
# ------------------------------------------------------------------------------
from collections import defaultdict, deque
import numpy as np
import os
import time
import torch
import tqdm

from habitat import Config, logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    linear_decay,
)
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, List, Optional

from habitat_extensions.utils import observations_to_image
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import transform_obs
from vlnce_baselines.agents.policy_cma_global import GlobalCMAPolicy
from vlnce_baselines.agents.policy_seq2seq_global import GlobalSeq2SeqPolicy

SUPPORTED_POLICIES = ["seq2seq"]

@baseline_registry.register_trainer(name="vlnce_ppo")
class VLNCEPPOTrainer(PPOTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["NavigatorTask-v0", "WaypointTask-v0"]

    def __init__(self, config=None):
        self.config = config
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")

        self._static_encoder = False
        self._encoder = None

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.
        Args:
            ppo_cfg: config node with relevant params
        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        model_cfg = self.config.MODEL 
        model_cfg.defrost()
        model_cfg.TORCH_GPU_ID = self.config.TORCH_GPU_ID
        model_cfg.freeze()
        
        assert model_cfg.POLICY in SUPPORTED_POLICIES, \
            f"{model_cfg.POLICY} not in {SUPPORTED_POLICIES}"
        
        if model_cfg.POLICY == "seq2seq":
            self.actor_critic = GlobalSeq2SeqPolicy(
                observation_space=self.envs.observation_spaces[0],
                action_space=self.envs.action_spaces[0],
                model_config=self.config.MODEL,
            )
        else:
            self.actor_critic = GlobalSeq2SeqPolicy(
                observation_space=self.envs.observation_spaces[0],
                action_space=self.envs.action_spaces[0],
                model_config=self.config.MODEL,
            )
            
        self.actor_critic.to(self.device)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )
        
        if self.config.LOAD_FROM_CKPT:
            ckpt_dict = self.load_checkpoint(
                self.config.LOAD_CKPT_FROM, map_location="cpu"
            )
            self.actor_critic.load_state_dict(ckpt_dict["state_dict_ac"])
            self.agent.load_state_dict(ckpt_dict["state_dict_agent"])
            logger.info(f"Loaded weights from checkpoint: {self.config.LOAD_CKPT_FROM}")
        logger.info("Finished setting up actor critic model.")

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.
        Args:
            file_name: file name for checkpoint
        Returns:
            None
        """
        checkpoint = {
            "state_dict_agent": self.agent.state_dict(),
            "state_dict_ac": self.actor_critic.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.
        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args
        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        
        if "instruction" in observations[0]:
            observations = transform_obs(
                observations, 
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
            )
        
        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        rewards = torch.tensor(
            rewards, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks
        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v

        current_episode_reward *= masks

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def train(self) -> None:
        r"""Main method for training PPO.
        Returns:
            None
        """
        # self.config.defrost()
        # self.config.SENSORS.append('SEMANTIC_SENSOR')
        # self.config.freeze()
        
        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )
        
        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )
        
        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        if "instruction" in observations[0]:
            observations = transform_obs(
                observations, 
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
            )
        batch = batch_obs(observations, device=self.device)
        
        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )
        
        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        
        count_checkpoints = 0
        update_start = 0
        if self.config.LOAD_FROM_CKPT:
            count_checkpoints = int(self.config.LOAD_CKPT_FROM.split(".")[1]) + 1
            update_start = (count_checkpoints) * self.config.CHECKPOINT_INTERVAL + 1
            logger.info(
                "Starting from ckpt: {}, update: {}".format(
                    count_checkpoints, update_start
                )
            )

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(update_start, self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                for step in range(ppo_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
                        rollouts, 
                        current_episode_reward, 
                        running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent(ppo_cfg, rollouts)
                pth_time += delta_pth_time

                for k, v in running_episode_stats.items():
                    window_episode_stats[k].append(v.clone())

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "reward", deltas["reward"] / deltas["count"], count_steps
                )

                # Check to see if there are any metrics
                # that haven't been logged yet
                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                if len(metrics) > 0:
                    writer.add_scalars("metrics", metrics, count_steps)

                losses = [value_loss, action_loss]
                writer.add_scalars(
                    "losses",
                    {k: l for l, k in zip(losses, ["value", "policy"])},
                    count_steps,
                )

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    logger.info(f"saved ckpt {count_checkpoints}")
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth", dict(step=count_steps)
                    )
                    count_checkpoints += 1

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.
        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging
        Returns:
            None
        """
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        # logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict_agent"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        if "instruction" in observations[0]:
            observations = transform_obs(
                observations, 
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
            )
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.EVAL.EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)

            outputs = self.envs.step([a[0].item() for a in actions])
            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            if "instruction" in observations[0]:
                observations = transform_obs(
                    observations, 
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
                )
            batch = batch_obs(observations, device=self.device)
            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()