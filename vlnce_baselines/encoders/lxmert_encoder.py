from pickle import TRUE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
import torchvision.models as models

from habitat_baselines.common.utils import Flatten

from vlnce_baselines import lxmert
from vlnce_baselines.lxmert.modeling_lxmert import LxmertModel, LxmertConfig
from vlnce_baselines.lxmert import utils, processing_image
from vlnce_baselines.lxmert.modeling_frcnn import GeneralizedRCNN

class LXMERTEncoder(nn.Module):
    def __init__(
        self,
        lxmert_config,
        frcnn_config,
        device = "cuda:0",
        spatial_output = False
    ):
        super().__init__()
        self.device = device
        
        # ----------------------------------------------------------------------
        # LXMERT 
        self.lxmert_config = LxmertConfig.from_pretrained(lxmert_config.config)
        self.lxmert = LxmertModel.from_pretrained(lxmert_config.model)
        for param in self.lxmert.parameters():
            param.requires_grad_(lxmert_config.is_trainable)
            
        self.use_pooled = lxmert_config.use_pooled
        self.language_len = 1 if self.use_pooled else lxmert_config.lng_len
        
        self._hidden_size = self.lxmert_config.hidden_size
        self._lang_output_size = lxmert_config.lng_output_size
        self._vis_output_size = lxmert_config.vis_output_size
        
        # ----------------------------------------------------------------------
        # FRCNN
        self.frcnn_cfg = utils.Config.from_pretrained(frcnn_config.config)
        self.bboxes = lxmert_config.vis_len 
        self.frcnn_cfg.min_detections = self.bboxes 
        self.frcnn_cfg.max_detections = self.bboxes
        
        self.image_preprocess = processing_image.Preprocess(self.frcnn_cfg)
        self.frcnn = GeneralizedRCNN.from_pretrained(frcnn_config.model)
        for param in self.frcnn.parameters():
            param.requires_grad_(frcnn_config.is_trainable)
        
        # ----------------------------------------------------------------------
        
        self.spatial_output = spatial_output
        if not self.spatial_output:
            # Linear layers to down-project the visual outputs
            self.vis_output_shape = (self.vis_output_size, )
            self.visual_linear = nn.Sequential(
                Flatten(),
                nn.Linear(
                    in_features=self.lxmert_visual_output_size,
                    out_features=self.vis_output_size
                ),
                nn.ReLU(True)
            )
        else:
            # Spatial layers to down-project the visual and language outputs
            # TODO: add spatial embedding
            pass
        
        # Linear layers to down-project the language outputs
        self.lang_output_shape = (self.lang_output_size, )
        self.language_linear = nn.Sequential(
            Flatten(),
            nn.Linear(
                in_features=self.lxmert_language_output_size, 
                out_features=self.lang_output_size
            ),
            nn.ReLU(True)
        )
        
    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def lxmert_language_output_size(self) -> int:
        return self.hidden_size * self.language_len
        
    @property
    def lxmert_visual_output_size(self) -> int:
        return self.hidden_size * self.bboxes
    
    @property
    def lang_output_size(self) -> int:
        return self._lang_output_size
        
    @property
    def vis_output_size(self) -> int:
        return self._vis_output_size
        

    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        is_lxmert_precomputed = (
            "lxmert_lang" in observations and "lxmert_vision" in observations
        )
        
        if is_lxmert_precomputed:
            visual_embedding = observations["lxmert_vision"].to(self.device)
            # visual_embedding = observations["lxmert_vision"].to(self.device)
            # visual_embedding = visual_embedding.reshape(
            #     visual_embedding.shape[0], -1)
            
            lang_embedding = observations["lxmert_lang"].unsqueeze(1)
            # lang_embedding = observations["lxmert_lang"].to(self.device)
            # lang_embedding = lang_embedding.reshape(
            #     lang_embedding.shape[0], -1)
            # print(f"vis: {visual_embedding.shape}")
        else:
            
            images, self.sizes, self.scales_xy  = self.image_preprocess(
                list(torch.unbind(observations["rgb"].to(self.device)))
            )
            images = images.to(self.device)
            
            output_dict = self.frcnn(
                images, 
                self.sizes, 
                scales_yx=self.scales_xy, 
                padding="max_detections",
                max_detections=self.frcnn_cfg.max_detections,
                return_tensors="pt",
            )
            
            normalized_boxes = output_dict.get("normalized_boxes").to(self.device)
            features = output_dict.get("roi_features").to(self.device)
            output = self.lxmert(
                input_ids=observations['input_ids'].squeeze(0),
                attention_mask=observations['attention_mask'].squeeze(0),
                visual_feats=features,
                visual_pos=normalized_boxes,
                token_type_ids=observations['token_type_ids'].squeeze(0),
                return_dict=True,
                output_attentions=False,
            )
            
            visual_embedding = output.vision_output
            # visual_embedding = output.vision_output.reshape(1, -1)
            
            if self.use_pooled:
                lang_embedding = output.pooled_output
            else:
                lang_embedding = output.language_output
            # lang_embedding = lang_embedding.reshape(1, -1)
            # print(f"noprec vis: {visual_embedding.shape}")
        
        lang_embedding = self.language_linear(lang_embedding)
        if self.spatial_output:
            # TODO: learn a spatial embedding 
            # shape after swap (batch, 768, bbox)
            visual_embedding = visual_embedding.swapaxes(1, 2)
            
            # shape after adding a dimension (batch, 768, 1)
            lang_embedding = lang_embedding.unsqueeze(2)
        else:
            visual_embedding = self.visual_linear(visual_embedding)
        
        return visual_embedding, lang_embedding