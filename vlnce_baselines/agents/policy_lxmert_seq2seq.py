import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat import Config
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.ppo.policy import Net

from vlnce_baselines.agents.policy import BasePolicy 
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.encoders.resnet_encoders import VlnResnetDepthEncoder
from vlnce_baselines.encoders.lxmert_encoder import LXMERTEncoder
from vlnce_baselines.encoders.simple_cnns import SimpleDepthCNN

class GlobalLXMERTSeq2seqPolicy(BasePolicy):
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
        self.model_config = model_config
        
        self.device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available() and not model_config.USE_CPU
            else torch.device("cpu")
        )
        
        # ----------------------------------------------------------------------
        # DEPTH 
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "SimpleDepthCNN",
            "VlnResnetDepthEncoder",
        ], "DEPTH_ENCODER.cnn_type must be SimpleDepthCNN or VlnResnetDepthEncoder"
        if model_config.DEPTH_ENCODER.cnn_type == "SimpleDepthCNN":
            self.depth_encoder = SimpleDepthCNN(
                observation_space, model_config.DEPTH_ENCODER.output_size
            )
        elif model_config.DEPTH_ENCODER.cnn_type == "VlnResnetDepthEncoder":
            self.depth_encoder = VlnResnetDepthEncoder(
                observation_space,
                output_size=model_config.DEPTH_ENCODER.output_size,
                checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
                backbone=model_config.DEPTH_ENCODER.backbone,
            )
            
        
        # ----------------------------------------------------------------------
        # LXMERT 
        self.lxmert_encoder = LXMERTEncoder(
            lxmert_config=model_config.LXMERT,
            frcnn_config=model_config.FRCNN_ENCODER,
            device=self.device
        )
        
        # ----------------------------------------------------------------------
        # RNN state encoder 
        rnn_input_size = (
            model_config.DEPTH_ENCODER.output_size +
            self.lxmert_encoder.lang_output_size +
            self.lxmert_encoder.vis_output_size
        )
        
        # PREV. ACTIONS
        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)
            rnn_input_size += self.prev_action_embedding.embedding_dim
            
        self.state_encoder = RNNStateEncoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            num_layers=1,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
        )
        
        self.progress_monitor = nn.Linear(
            self.model_config.STATE_ENCODER.hidden_size, 1
        )

        self._init_layers()
        
        self.train()

    @property
    def output_size(self):
        return self.model_config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _init_layers(self):
        nn.init.kaiming_normal_(self.progress_monitor.weight, nonlinearity="tanh")
        nn.init.constant_(self.progress_monitor.bias, 0)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        
        rgb_embedding, instruction_embedding = self.lxmert_encoder(observations)
        
        depth_embedding = self.depth_encoder(observations)
        
        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
            
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
            
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0
            
        x = torch.cat(
            [instruction_embedding, depth_embedding, rgb_embedding],
            dim=1
        )
    
        if self.model_config.SEQ2SEQ.use_prev_action:
            prev_actions_embedding = self.prev_action_embedding(
                ((prev_actions.float() + 1) * masks).long().view(-1)
            )
            x = torch.cat([x, prev_actions_embedding], dim=1)

        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        if self.model_config.PROGRESS_MONITOR.use and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(x))
            progress_loss = F.mse_loss(
                progress_hat.squeeze(1), 
                observations["progress"], 
                reduction="none"
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss,
                self.model_config.PROGRESS_MONITOR.alpha,
            )

        return x, rnn_hidden_states
