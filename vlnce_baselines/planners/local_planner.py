# ------------------------------------------------------------------------------
# @file: local_planner.py
# @brief: Given a goal and image features executes a plan to the goal.
# ------------------------------------------------------------------------------
from sys import path
import numpy as np
from habitat.utils.visualizations.utils import images_to_video
import torch
import os

from gym import Space
from habitat import logger
from habitat.config import Config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim

from habitat_baselines.rl.ppo import PPO
from habitat_baselines.utils.common import Flatten, batch_obs
from habitat_baselines.utils.common import batch_obs

from typing import Optional, Union

from habitat_extensions.utils import resize, observations_to_image
from vlnce_baselines.agents.policy_seq2seq_local import LocalSeq2SeqPolicy

SUPPORTED_POLICIES = ["seq2seq", "cma"]

class LocalPlanner():
    def __init__(self,
                 config: Config, observation_space: Space, action_space: Space
                 ):
        r''' Configures the local planner.
        Args:
            -config: YAML file containing all configuration information needed
            -observation_space: input Space of the local planner
            -action_space: output Space of the local planner. 
        '''
        self.config = config
        self.keep_frames = self.config.keep_frames
        self.observation_space = observation_space
        self.action_space = action_space
        self.nproc = config.n_processes
        self.frames = []
        self.device = (
            torch.device("cuda", self.config.torch_gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.build_planner()
        logger.info(f"Local planner is ready!")

    def build_planner(self):
        model_cfg = self.config.MODEL
        model_cfg.defrost()
        model_cfg.TORCH_GPU_ID = self.config.torch_gpu_id
        model_cfg.freeze()

        assert model_cfg.POLICY in SUPPORTED_POLICIES, \
            f"{model_cfg.POLICY} not in {SUPPORTED_POLICIES}"

        if model_cfg.POLICY == "seq2seq":
            self.actor_critic = LocalSeq2SeqPolicy(
                observation_space=self.observation_space,
                action_space=self.action_space,
                model_config=model_cfg,
            )
        else:
            logger.error(f"invalid policy {model_cfg.POLICY}")
            raise ValueError

        self.actor_critic.to(self.device)

        self.ppo = self.config.RL.PPO
        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=self.ppo.clip_param,
            ppo_epoch=self.ppo.ppo_epoch,
            num_mini_batch=self.ppo.num_mini_batch,
            value_loss_coef=self.ppo.value_loss_coef,
            entropy_coef=self.ppo.entropy_coef,
            lr=self.ppo.lr,
            eps=self.ppo.eps,
            max_grad_norm=self.ppo.max_grad_norm,
            use_normalized_advantage=self.ppo.use_normalized_advantage,
        )
        self.recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.n_processes,
            self.ppo.hidden_size,
            device=self.device
        )

        logger.info(f"Loading weights from: {self.ppo.checkpoint}")
        ckpt_dict = torch.load(self.ppo.checkpoint, map_location="cpu")
        # # self.actor_critic.load_state_dict(ckpt_dict["state_dict_ac"])
        self.agent.load_state_dict(ckpt_dict["state_dict_agent"])
        self.actor_critic = self.agent.actor_critic
        self.reset()

    def reset(self) -> None:
        self.recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.n_processes,
            self.ppo.hidden_size,
            device=self.device
        )

        self.prev_action = torch.zeros(
            self.nproc, 1, device=self.device, dtype=torch.long)

        self.not_done_masks = torch.zeros(
            self.nproc, 1, device=self.device)

        self.is_done = False
       

    def run(self, observation, waypoint, env):
       
        self.reset()
        observation[0]["pointgoal"] = waypoint
        
        if self.keep_frames:
            f = np.concatenate(
                (observation[0]["rgb"], observation[0]["rgb"]), axis=1)
            self.frames.append(f)
            self.frames.append(f)

        while not self.is_done and not env.habitat_env.episode_over:
            action = self.get_next_action(observation, True) # this is going to sample actions
            print(f"action: {action}")

            if action == HabitatSimActions.STOP:
                self.is_done = True
                dones = [True]
            else:
                observations = [env.step(action)]
                observations, rewards, dones, infos = [
                    list(x) for x in zip(*observations)
                ]
                
                observation = [{
                    "rgb": observations[0]["rgb"],
                    "depth": observations[0]["depth"],
                    "pointgoal": waypoint
                }]
                
                if self.keep_frames:
                    frame = observations_to_image(observation[0], infos[0])
                    self.frames.append(frame)
                    
                self.update_masks(dones)
        
        if self.keep_frames:
            self.generate_output(
                resize(self.frames), 
                "path_id={}".format(env.current_episode.episode_id)
            )
                        
        return observation
    
    def set_path(self, path):
        self.path = path
        
    def generate_output(self, frames, filename):
        images_to_video(frames,  self.path, filename)
        self.frames = []

    def get_actor_critic(self):
        return self.actor_critic

    def get_agent(self):
        return self.agent

    def get_next_action(
        self, observations, deterministic: Optional[bool] = False
    ) -> int:
        batch = batch_obs(observations, device=self.device)
        self.actor_critic.eval()

        with torch.no_grad():
            (
                _,
                action,
                _,
                self.recurrent_hidden_states,
            ) = self.actor_critic.act(
                batch,
                self.recurrent_hidden_states,
                self.prev_action,
                self.not_done_masks,
                deterministic=deterministic,
            )
            self.prev_action = action
        return action.item()

    def update_masks(self, dones) -> None:
        self.not_done_masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=self.device,
        )
