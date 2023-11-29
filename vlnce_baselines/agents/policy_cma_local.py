import abc
from pickle import TRUE

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from habitat import Config
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Net

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.encoders.instruction_encoder import InstructionEncoder
from vlnce_baselines.encoders.rcm_state_encoder import RCMStateEncoder
from vlnce_baselines.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
)
from vlnce_baselines.agents.policy import BasePolicy 

class LocalCMAPolicy(BasePolicy):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config
    ):
        super().__init__(
            CMANet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )


class CMANet(Net):
    r""" A cross-modal attention (CMA) network that contains:
        Depth encoder
        RGB encoder
        RNN state encoder or CMA state encoder
    """

    def __init__(self, observation_space: Space, model_config: Config, num_actions):
        super().__init__()
        self.model_config = model_config
        model_config.defrost()
        model_config.INSTRUCTION_ENCODER.final_state_only = False
        model_config.freeze()

        self.device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder"
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=True,
        )

        # Init the RGB encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "TorchVisionResNet50"
        ], "RGB_ENCODER.cnn_type must be TorchVisionResNet50'."

        self.rgb_encoder = TorchVisionResNet50(
            observation_space,
            model_config.RGB_ENCODER.output_size,
            self.device,
            spatial_output=True,
        )
        
        hidden_size = model_config.STATE_ENCODER.hidden_size
        self._hidden_size = hidden_size
        
        self.rgb_kv = nn.Conv1d(
            self.rgb_encoder.output_shape[0],
            hidden_size // 2 + model_config.RGB_ENCODER.output_size,
            1,
        )

        self.depth_kv = nn.Conv1d(
            self.depth_encoder.output_shape[0],
            hidden_size // 2 + model_config.RGB_ENCODER.output_size,
            1,
        )
        
        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                model_config.STATE_ENCODER.hidden_size * 16, # spatial output is 4x4
                model_config.RGB_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )
        
        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                model_config.STATE_ENCODER.hidden_size * 16, # spatial output is 4x4
                model_config.DEPTH_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )

        # Init the RNN state decoder
        rnn_input_size = model_config.DEPTH_ENCODER.output_size
        rnn_input_size += model_config.RGB_ENCODER.output_size
        rnn_input_size += self.prev_action_embedding.embedding_dim
        
        if "pointgoal_with_gps_compass" in observation_space.spaces:
            rnn_input_size += observation_space.spaces["pointgoal_with_gps_compass"].shape[0]
            
        if "pointgoal" in observation_space.spaces:
            rnn_input_size += observation_space.spaces["pointgoal"].shape[0]
            
        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            num_layers=1,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )

        self.register_buffer(
            "_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5)))

        self._output_size = model_config.STATE_ENCODER.hidden_size

        self.train()

    @property
    def output_size(self):
        return self._output_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers 
        
    def _coattn(self, d, k, v):
        logits = torch.einsum("nci, ndi -> ncd", d, k)
        attn = F.softmax(logits * self._scale, dim=1)
        return torch.einsum("ncd, ndi -> nci", attn, v)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        
        # ----------------------------------------------------------------------
        # Encode visual inputs 
        # shape (batch, 192, 4, 4)
        depth_embedding = self.depth_encoder(observations)
        # shape (batch, 192, 16)
        depth_embedding = torch.flatten(depth_embedding, 2)
        
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0.0
        
        # shape (batch, 2112, 4, 4)
        rgb_embedding = self.rgb_encoder(observations)
        # shape (batch, 2112, 16)
        rgb_embedding = torch.flatten(rgb_embedding, 2)
        
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0.0
        
        # ----------------------------------------------------------------------
        # Co-attention
        # depth_kv shape (batch, 384, 16)
        depth_kv = self.depth_kv(depth_embedding)
        # depth_k shape (batch, 256, 16)
        # depth_v shape (batch, 128, 16)
        depth_k, depth_v = torch.split(depth_kv, self._hidden_size // 2, dim=1)
        
        # # rgb_kv shape (batch, 512, 16)
        rgb_kv = self.rgb_kv(rgb_embedding)
        # rgb_k shape (batch, 256, 16)
        # rgb_v shape (batch, 256, 16)
        rgb_k, rgb_v = torch.split(rgb_kv, self._hidden_size // 2, dim=1)
        
        # co-attention 
        # shape (1, 512, 16)
        rgb_embedding = self._coattn(depth_kv, rgb_k, rgb_v)
        # shape (1, 512, 16)
        depth_embedding = self._coattn(rgb_kv, depth_k, depth_v)
        
        # ----------------------------------------------------------------------
        # State encoding 
        # shape (1, 128)
        depth_in = self.depth_linear(depth_embedding)
        
        # shape (1, 256)
        rgb_in = self.rgb_linear(rgb_embedding)
        
        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().view(-1)
        )
        
        pointgoal_encoding = torch.zeros(
            [1, 2], dtype=torch.float32, device=self.device)
        if "pointgoal_with_gps_compass" in observations:
            pointgoal_encoding = observations["pointgoal_with_gps_compass"]
        elif "pointgoal" in observations:
            pointgoal_encoding = observations["pointgoal"]
            
        if self.model_config.ablate_pointgoal:
            pointgoal_encoding = pointgoal_encoding * 0.0
            
        # shape (1, 418)
        x = torch.cat([rgb_in, depth_in, prev_actions, pointgoal_encoding], dim=1)
    
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        
        return x, rnn_hidden_states


if __name__ == "__main__":
    from vlnce_baselines.config.default import get_config
    from gym import spaces

    config = get_config("habitat_baselines/config/vln/il_vln.yaml")

    device = (
        torch.device("cuda", config.TORCH_GPU_ID)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    observation_space = spaces.Dict(
        dict(
            rgb=spaces.Box(low=0, high=0, shape=(224, 224, 3), dtype=np.float32),
            depth=spaces.Box(low=0, high=0, shape=(256, 256, 1), dtype=np.float32),
        )
    )

    # Add TORCH_GPU_ID to VLN config for a ResNet layer
    config.defrost()
    config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
    config.freeze()

    action_space = spaces.Discrete(4)

    policy = GlobalCMAPolicy(
        observation_space, action_space, config.MODEL
    ).to(device)

    dummy_instruction = torch.randint(1, 4, size=(4 * 2, 8), device=device)
    dummy_instruction[:, 5:] = 0
    dummy_instruction[0, 2:] = 0

    obs = dict(
        rgb=torch.randn(4 * 2, 224, 224, 3, device=device),
        depth=torch.randn(4 * 2, 256, 256, 1, device=device),
        instruction=dummy_instruction,
        progress=torch.randn(4 * 2, 1, device=device),
    )

    hidden_states = torch.randn(
        policy.net.state_encoder.num_recurrent_layers,
        2,
        policy.net._hidden_size,
        device=device,
    )
    prev_actions = torch.randint(0, 3, size=(4 * 2, 1), device=device)
    masks = torch.ones(4 * 2, 1, device=device)

    AuxLosses.activate()

    policy.evaluate_actions(obs, hidden_states, prev_actions, masks, prev_actions)
