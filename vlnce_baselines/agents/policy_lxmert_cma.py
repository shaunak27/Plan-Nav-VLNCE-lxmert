import abc
from shutil import ExecError
from time import time
from filelock import logger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat import Config
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Net

from vlnce_baselines.agents.policy import BasePolicy 
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.encoders.lxmert_encoder import LXMERTEncoder
from vlnce_baselines.encoders.resnet_encoders import VlnResnetDepthEncoder

class GlobalLXMERTCMAPolicy(BasePolicy):
    def __init__(
        self, observation_space: Space, action_space: Space, model_config: Config
    ):
        super().__init__(
            LXMERTNet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )


class LXMERTNet(Net):
    r"""A baseline sequence to sequence network that concatenates instruction,
    RGB, and depth encodings before decoding an action distribution with an RNN.

    Modules:
        LXMERT encoder
        Depth encoder
        RNN state encoder
    """
    def __init__(self, observation_space: Space, model_config: Config, num_actions):
        super().__init__()
        self.config = model_config
        
        self.device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available() and not model_config.USE_CPU
            else torch.device("cpu")
        )
        
        self._hidden_size = model_config.STATE_ENCODER.hidden_size
        
        # ----------------------------------------------------------------------
        # DEPTH
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder",
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
            
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=True
        )
        
        # ----------------------------------------------------------------------
        # LXMERT 
        self.lxmert_encoder = LXMERTEncoder(
            lxmert_config=model_config.LXMERT,
            frcnn_config=model_config.FRCNN_ENCODER,
            device=self.device,
            spatial_output=True
        )
        
        # ----------------------------------------------------------------------
        # PREV. ACTIONS
        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
        
        # ----------------------------------------------------------------------
        # CROSS-MODAL ATTENTION 
        
        # NOTE: I tried to reduce tensor sizes as much as possible for the RNNs
        rnn_input_size = (
            self.prev_action_embedding.embedding_dim
            + model_config.RGB_ENCODER.output_size
            + model_config.DEPTH_ENCODER.output_size 
        )
       
        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                self.lxmert_encoder.hidden_size,
                model_config.RGB_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )
        
        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod(self.depth_encoder.output_shape),
                model_config.DEPTH_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )

        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=self._hidden_size,
            num_layers=1,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )
        
        self.rgb_kv = nn.Conv1d(
            self.lxmert_encoder.hidden_size,
            self._hidden_size // 2 + model_config.RGB_ENCODER.output_size,
            1,
        )
        
        self.depth_kv = nn.Conv1d(
            self.depth_encoder.output_shape[0],
            self._hidden_size // 2 + model_config.DEPTH_ENCODER.output_size,
            1,
        )
        
        self.state_q = nn.Linear(
            self._hidden_size, 
            self._hidden_size // 2
        )
        
        self.text_k = nn.Conv1d(
            self.lxmert_encoder.lang_output_size,
            self._hidden_size // 2,
            1
        )
        
        self.text_q = nn.Linear(
            self.lxmert_encoder.lang_output_size, 
            self._hidden_size // 2
        )

        self.register_buffer(
            "_scale", 
            torch.tensor(1.0 / ((self.lxmert_encoder.hidden_size // 2) ** 0.5))
        )
        
        self._output_size = (
            self._hidden_size
            + model_config.RGB_ENCODER.output_size
            + model_config.DEPTH_ENCODER.output_size 
            + self.lxmert_encoder.lang_output_size
            + self.prev_action_embedding.embedding_dim
        )

        self.second_state_compress = nn.Sequential(
            nn.Linear(
                self._output_size,
                self._hidden_size,
            ),
            nn.ReLU(True),
        )
        
        self.second_state_encoder = RNNStateEncoder(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            num_layers=1,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )
        
        self._output_size = model_config.STATE_ENCODER.hidden_size
        
        self.progress_monitor = nn.Linear(self.output_size, 1)

        self._init_layers()

        self.train()

    @property
    def output_size(self):
        return self.config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers + (
            self.second_state_encoder.num_recurrent_layers
        )

    def _init_layers(self):
        nn.init.kaiming_normal_(self.progress_monitor.weight, nonlinearity="tanh")
        nn.init.constant_(self.progress_monitor.bias, 0)
    
    def _attn(self, q, k, v, mask=None):
        logits = torch.einsum("nc, nci -> ni", q, k)
        
        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)
        
        return torch.einsum("ni, nci -> nc", attn, v)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        
        rgb_embedding, instruction_embedding = self.lxmert_encoder(observations)
        depth_embedding = self.depth_encoder(observations)
        
        # shape (1, 32)
        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().view(-1)
        )
        
        # shape after linear (1, 256)
        rgb_in = self.rgb_linear(rgb_embedding)
        
        # shape after flatten (1, 192, 16)
        depth_embedding = self.depth_encoder(observations)
        depth_embedding = torch.flatten(depth_embedding, 2)
        # depth (1, 128)
        depth_in = self.depth_linear(depth_embedding)
        
        # state_in (1, rnn_input_size)
        state_in = torch.cat([rgb_in, depth_in, prev_actions], dim=1)
            
        # state (1, STATE ENCODER.hidden_size = 512)
        (
            state,
            rnn_hidden_states[0 : self.state_encoder.num_recurrent_layers],
        ) = self.state_encoder(
            state_in,
            rnn_hidden_states[0 : self.state_encoder.num_recurrent_layers],
            masks,
        )
        
        # Attention over the state + language embedding
        # shape (1, 256)
        text_state_q = self.state_q(state)
        
        # shape (1, 256, 1)
        text_state_k = self.text_k(instruction_embedding)
        
        # shape (1, 1)
        text_mask = (instruction_embedding == 0.0).all(dim=1)
        
        # shape (1, 768)
        text_embedding = self._attn(
            text_state_q, text_state_k, instruction_embedding, text_mask
        )
        
        # Attention over the attended language ^^ and the rgb embedding
        # shape (1, 256)
        text_q = self.text_q(text_embedding)
        # rgb_k shape (1, 256, bbox)
        # rgb_v shape (1, 128, bbox)
        rgb_k, rgb_v = torch.split(
            self.rgb_kv(rgb_embedding), 
            self._hidden_size // 2, 
            dim=1
        )
        # shape (1, 256)
        rgb_embedding = self._attn(text_q, rgb_k, rgb_v)
        
        # Attention over the attended language ^^ and the depth embedding
        # depth_k shape (1, 256, 16)
        # depth_v shape (1, 128, 16)
        depth_k, depth_v = torch.split(
            self.depth_kv(depth_embedding), 
            self._hidden_size // 2, 
            dim=1
        )
        # shape (1, 128)
        depth_embedding = self._attn(text_q, depth_k, depth_v)
        
        # Second state encoding pass
        # shape (1, out_hidden_State)
        x = torch.cat(
            [state, text_embedding, rgb_embedding, depth_embedding, prev_actions], dim=1
        )
        
        # shape (1, 512)
        x = self.second_state_compress(x)
        (
            x,
            rnn_hidden_states[self.state_encoder.num_recurrent_layers :],
        ) = self.second_state_encoder(
            x, 
            rnn_hidden_states[self.state_encoder.num_recurrent_layers :], 
            masks
        )

        if self.config.PROGRESS_MONITOR.use and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(x))
            progress_loss = F.mse_loss(
                progress_hat.squeeze(1), 
                observations["progress"], 
                reduction="none"
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss,
                self.config.PROGRESS_MONITOR.alpha,
            )

        return x, rnn_hidden_states
