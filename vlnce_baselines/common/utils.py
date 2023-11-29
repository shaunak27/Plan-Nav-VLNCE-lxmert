from typing import Dict, List, Optional

from collections import defaultdict
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as f
from habitat.utils.visualizations import maps
import gzip,json
import torch
import torch.nn as nn
from vlnce_baselines.agents.r2r_envdrop.utils import Tokenizer, read_vocab
from habitat.utils.visualizations.utils import images_to_video
from scipy.io import wavfile
from moviepy.audio.AudioClip import CompositeAudioClip
import os
from collections import defaultdict
from typing import Dict, List, Optional, Any
import random
import json

import moviepy.editor as mpy
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import draw_collision
from torch.utils.tensorboard import SummaryWriter

class TensorboardWriter:
    def __init__(self, log_dir: str, *args: Any, **kwargs: Any):
        r"""A Wrapper for tensorboard SummaryWriter. It creates a dummy writer
        when log_dir is empty string or None. It also has functionality that
        generates tb video directly from numpy images.

        Args:
            log_dir: Save directory location. Will not write to disk if
            log_dir is an empty string.
            *args: Additional positional args for SummaryWriter
            **kwargs: Additional keyword args for SummaryWriter
        """
        self.writer = None
        if log_dir is not None and len(log_dir) > 0:
            self.writer = SummaryWriter(log_dir, *args, **kwargs)

    def __getattr__(self, item):
        if self.writer:
            return self.writer.__getattribute__(item)
        else:
            return lambda *args, **kwargs: None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.close()

    def add_video_from_np_images(
        self, video_name: str, step_idx: int, images: np.ndarray, fps: int = 10
    ) -> None:
        r"""Write video into tensorboard from images frames.

        Args:
            video_name: name of video string.
            step_idx: int of checkpoint index to be displayed.
            images: list of n frames. Each frame is a np.ndarray of shape.
            fps: frame per second for output video.

        Returns:
            None.
        """
        if not self.writer:
            return
        # initial shape of np.ndarray list: N * (H, W, 3)
        frame_tensors = [
            torch.from_numpy(np_arr).unsqueeze(0) for np_arr in images
        ]
        video_tensor = torch.cat(tuple(frame_tensors))
        video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)
        # final shape of video tensor: (1, n, 3, H, W)
        self.writer.add_video(
            video_name, video_tensor, fps=fps, global_step=step_idx
        )


class Flatten(nn.Module):
    def forward(self, x):
        # return x.view(x.size(0), -1)
        return x.reshape(x.size(0), -1)
    
    
class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

class CategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)


class CategoricalNetWithMask(nn.Module):
    def __init__(self, num_inputs, num_outputs, masking):
        super().__init__()
        self.masking = masking

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, features, action_maps):
        probs = f.softmax(self.linear(features))
        if self.masking:
            probs = probs * torch.reshape(
                action_maps, (action_maps.shape[0], -1)).float()

        return CustomFixedCategorical(probs=probs)


def linear_decay(epoch: int, total_num_updates: int) -> float:
    r"""Returns a multiplicative factor for linear value decay
    Args:
        epoch: current epoch number
        total_num_updates: total number of epochs
    Returns:
        multiplicative factor that decreases param value linearly
    """
    return 1 - (epoch / float(total_num_updates))

def transform_obs(
    observations: List[Dict], instruction_sensor_uuid: str
) -> Dict[str, torch.Tensor]:
    r"""Extracts instruction tokens from an instruction sensor and
    transposes a batch of observation dicts to a dict of batched
    observations.
    Args:
        observations:  list of dicts of observations.
        instruction_sensor_uuid: name of the instructoin sensor to
            extract from.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None
    Returns:
        transposed dict of lists of observations.
    """
    for i in range(len(observations)):
        observations[i][instruction_sensor_uuid] = observations[i][
            instruction_sensor_uuid
        ]["tokens"]
    return observations

def _to_tensor(v) -> torch.Tensor:
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)
    
def batch_obs(
    observations: List[Dict], device: Optional[torch.device] = None, skip_list = []
) -> Dict[str, torch.Tensor]:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of lists of observations.
    """
    batch = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            if sensor in skip_list:
                continue
            if sensor not in ['input_ids', 'token_type_ids', 'attention_mask','rgb']:
                batch[sensor].append(_to_tensor(obs[sensor]).float())
            else :
                batch[sensor].append(_to_tensor(obs[sensor]))

    for sensor in batch:
        if sensor not in ['input_ids', 'token_type_ids', 'attention_mask','rgb']:
            batch[sensor] = (
                torch.stack(batch[sensor], dim=0)
                .to(device=device)
                .to(dtype=torch.float)
            )
        else :
            batch[sensor] = (
                torch.stack(batch[sensor], dim=0)
                .to(device=device)
            )

    return batch

def exponential_decay(epoch: int, total_num_updates: int, decay_lambda: float) -> float:
    r"""Returns a multiplicative factor for linear value decay
    Args:
        epoch: current epoch number
        total_num_updates: total number of epochs
        decay_lambda: decay lambda
    Returns:
        multiplicative factor that decreases param value linearly
    """
    return np.exp(-decay_lambda * (epoch / float(total_num_updates)))


def plot_top_down_map(info, dataset='replica', pred=None):
    top_down_map = info["top_down_map"]["map"]
    top_down_map = maps.colorize_topdown_map(
        top_down_map, info["top_down_map"]["fog_of_war_mask"]
    )
    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    if dataset == 'replica':
        agent_radius_px = top_down_map.shape[0] // 16
    else:
        agent_radius_px = top_down_map.shape[0] // 50
    top_down_map = maps.draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,
        agent_rotation=info["top_down_map"]["agent_angle"],
        agent_radius_px=agent_radius_px
    )
    if pred is not None:
        from habitat.utils.geometry_utils import quaternion_rotate_vector

        source_rotation = info["top_down_map"]["agent_rotation"]

        rounded_pred = np.round(pred[1])
        direction_vector_agent = np.array([rounded_pred[1], 0, -rounded_pred[0]])
        direction_vector = quaternion_rotate_vector(source_rotation, direction_vector_agent)

        grid_size = (
            (maps.COORDINATE_MAX - maps.COORDINATE_MIN) / 10000,
            (maps.COORDINATE_MAX - maps.COORDINATE_MIN) / 10000,
        )
        delta_x = int(-direction_vector[0] / grid_size[0])
        delta_y = int(direction_vector[2] / grid_size[1])

        x = np.clip(map_agent_pos[0] + delta_x, a_min=0, a_max=top_down_map.shape[0])
        y = np.clip(map_agent_pos[1] + delta_y, a_min=0, a_max=top_down_map.shape[1])
        point_padding = 20
        for m in range(x - point_padding, x + point_padding + 1):
            for n in range(y - point_padding, y + point_padding + 1):
                if np.linalg.norm(np.array([m - x, n - y])) <= point_padding and \
                        0 <= m < top_down_map.shape[0] and 0 <= n < top_down_map.shape[1]:
                    top_down_map[m, n] = (0, 255, 255)
        if np.linalg.norm(rounded_pred) < 1:
            assert delta_x == 0 and delta_y == 0

    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)
    return top_down_map


def resize_observation(observations, model_resolution):
    for observation in observations:
        observation['rgb'] = cv2.resize(
            observation['rgb'], (model_resolution, model_resolution))
        observation['depth'] = np.expand_dims(cv2.resize(observation['depth'], (model_resolution, model_resolution)),
                                              axis=-1)
                          
def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    scene_name: str,
    episode_id: int,
    checkpoint_idx: int,
    metric_name: str,
    metric_value: float,
    fps: int = 10,
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
        audios: raw audio files
    Returns:
        None
    """
    if len(images) < 1:
        return

    video_name = f"{scene_name}_{episode_id}_{metric_name}{metric_value:.2f}"
    if "disk" in video_option:
        assert video_dir is not None
       
        images_to_video(images, video_dir, video_name)
