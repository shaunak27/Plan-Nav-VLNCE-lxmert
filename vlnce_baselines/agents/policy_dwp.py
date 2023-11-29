#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat.config.default import Config

from vlnce_baselines.agents.policy import BasePolicy
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.utils import CategoricalNetWithMask
from vlnce_baselines.encoders.instruction_encoder import InstructionEncoder
from vlnce_baselines.encoders.map_cnn import MapCNN
from vlnce_baselines.encoders.rnn_state_encoder import RNNStateEncoder
from vlnce_baselines.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
)
from vlnce_baselines.encoders.simple_cnns import SimpleDepthCNN, SimpleRGBCNN
from vlnce_baselines.encoders.visual_cnn import VisualCNN


DUAL_GOAL_DELIMITER = ','


class Policy(nn.Module):
    def __init__(self, net, dim_actions, masking=True):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNetWithMask(
            self.net.output_size, self.dim_actions, masking
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self, observations, rnn_hidden_states, prev_actions, masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        
        distribution = self.action_distribution(
            features, observations["action_map"]
        )
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, distribution

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        action_map = observations['action_map']
        distribution = self.action_distribution(features, action_map)
        value = self.critic(features)
        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class DWSeq2SeqPolicy(Policy):
    def __init__(
        self, observation_space: Space, action_space: Space, action_map_size: int, 
        model_config: Config, masking=True
    ):
        super().__init__(
            DWSeq2SeqNet(
                observation_space=observation_space,
                model_config=model_config,
            ),
            dim_actions=action_map_size ** 2,
            masking=masking
            # action_space.n
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class DWSeq2SeqNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    gm: geometric map
    """

    def __init__(self, observation_space: Space, model_config: Config): 
        super().__init__()

        spaces = observation_space.spaces
        self._depth = "depth" in spaces
        self._rgb = "rgb" in spaces
        self._instruction = "instruction" in spaces
        self._geometric_map = "gm" in spaces
        
        self._config = model_config
        
        self._hidden_size = model_config.STATE_ENCODER.hidden_size
        
        rnn_input_size = 0
        
        # self.visual_encoder = VisualCNN(
        #     observation_space, self._hidden_size, self._rgb, self._depth)
        # if not self.is_blind:
        #     rnn_input_size += self._hidden_size
        
        # ----------------------------------------------------------------------
        # depth encoding
        if self._depth:
            assert self._config.DEPTH_ENCODER.cnn_type in [
                "SimpleDepthCNN", "VlnResnetDepthEncoder" 
            ], "DEPTH_ENCODER.cnn_type must be SimpleDepthCNN or VlnResnetDepthEncoder"
        
            if self._config.DEPTH_ENCODER.cnn_type == "SimpleDepthCNN":
                self.depth_encoder = SimpleDepthCNN(
                    observation_space=observation_space, 
                    output_size=model_config.DEPTH_ENCODER.output_size
                )
            elif self._config.DEPTH_ENCODER.cnn_type == "VlnResnetDepthEncoder":
                self.depth_encoder = VlnResnetDepthEncoder(
                    observation_space=observation_space,
                    output_size=self._config.DEPTH_ENCODER.output_size,
                    checkpoint=self._config.DEPTH_ENCODER.ddppo_checkpoint,
                    backbone=self._config.DEPTH_ENCODER.backbone,
            )
                
            rnn_input_size += self._config.DEPTH_ENCODER.output_size
        
        # ----------------------------------------------------------------------
        # rgb encoding
        if self._rgb:
            assert self._config.RGB_ENCODER.cnn_type in [
                "SimpleRGBCNN", "TorchVisionResNet50",
            ], "RGB_ENCODER.cnn_type must be either 'SimpleRGBCNN' or 'TorchVisionResNet50'."

            if self._config.RGB_ENCODER.cnn_type == "SimpleRGBCNN":
                self.rgb_encoder = SimpleRGBCNN(
                    observation_space=observation_space, 
                    output_size=self._config.RGB_ENCODER.output_size
                )
            elif self._config.RGB_ENCODER.cnn_type == "TorchVisionResNet50":
                device = (
                    torch.device("cuda", self._config.TORCH_GPU_ID)
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                self.rgb_encoder = TorchVisionResNet50(
                    observation_space=observation_space,
                    output_size=self._config.RGB_ENCODER.output_size, 
                    device=device
                )
                
            rnn_input_size += self._config.RGB_ENCODER.output_size
        
        # ----------------------------------------------------------------------
        # instruction encoding
        if self._instruction:
            self.instruction_encoder = InstructionEncoder(
                config=self._config.INSTRUCTION_ENCODER
            )
            
            rnn_input_size += self.instruction_encoder.output_size
        
        # ----------------------------------------------------------------------
        # map encoding
        if self._geometric_map:
            self.geometric_map_encoder = MapCNN(
                observation_space, self._hidden_size, map_type='gm'
            )
            
            rnn_input_size += self._hidden_size
            
        # ----------------------------------------------------------------------   
        if self._config.SEQ2SEQ.use_prev_action:
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=self._hidden_size,
            num_layers=1,
            rnn_type=self._config.STATE_ENCODER.rnn_type,
        )

        self.progress_monitor = nn.Linear(
            self._config.STATE_ENCODER.hidden_size, 1
        )

        self._init_layers()
        self.train()

    @property
    def output_size(self):
        return self._hidden_size
    
    def _init_layers(self):
        nn.init.kaiming_normal_(self.progress_monitor.weight, nonlinearity="tanh")
        nn.init.constant_(self.progress_monitor.bias, 0)

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers
    
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = []
        
        if self._depth:
            depth_embedding = self.depth_encoder(observations)
            if self._config.ablate_depth:
                depth_embedding = depth_embedding * 0
            x.append(depth_embedding)
        
        if self._rgb:
            rgb_embedding = self.rgb_encoder(observations)
            if self._config.ablate_depth:
                rgb_embedding = rgb_embedding * 0
            x.append(rgb_embedding)
        
        # if not self.is_blind:
        #     visual_embedding = self.visual_encoder(observations)
        #     x.append(visual_embedding)
            
        if self._instruction:
            # TODO: Fix forward function
            instruction_embedding = self.instruction_encoder(observations)
            if self._config.ablate_instruction:
                instruction_embedding = instruction_embedding * 0
            x.append(instruction_embedding)
        
        if self._geometric_map:
            gmap_embedding = self.geometric_map_encoder(observations)
            if self._config.ablate_map:
                gmap_embedding = gmap_embedding * 0
            x.append(gmap_embedding)
        
        x = torch.cat(x, dim=1)
        
        if self._config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x = torch.cat([x, prev_actions_embedding], dim=1)
        
        x2, rnn_hidden_states1 = self.state_encoder(
            x, rnn_hidden_states, masks
        )
        
        if self._config.PROGRESS_MONITOR.use and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(x))
            progress_loss = F.mse_loss(
                progress_hat.squeeze(1), 
                observations["progress"], 
                reduction="none"
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss,
                self._config.PROGRESS_MONITOR.alpha,
            )

        assert not torch.isnan(x2).any().item()
        
        return x2, rnn_hidden_states1
