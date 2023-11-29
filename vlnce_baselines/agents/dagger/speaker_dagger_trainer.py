import gc
import json
import os
import random
from shutil import ExecError
import time
import warnings
from collections import defaultdict
from typing import Dict, List
import lmdb
import msgpack_numpy
import numpy as np
from tensorflow.python.keras.backend import backend
import torch
import torch.multiprocessing
import torch.nn.functional as F
import tqdm

from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import generate_video

from habitat_extensions.utils import observations_to_image
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.env_utils import (
    construct_envs,
    construct_envs_auto_reset_false,
)
from vlnce_baselines.common.utils import batch_obs
from vlnce_baselines.agents.policy_lxmert_seq2seq import GlobalLXMERTSeq2seqPolicy
from vlnce_baselines.agents.policy_lxmert_cma import GlobalLXMERTCMAPolicy

sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)
 
def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


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
        observations[i]['input_ids'] = observations[i][
            instruction_sensor_uuid
        ]['input_ids']
        observations[i]['token_type_ids'] = observations[i][
            instruction_sensor_uuid
        ]['token_type_ids']
        observations[i]['attention_mask'] = observations[i][
            instruction_sensor_uuid
        ]['attention_mask']
        # observations[i].pop(instruction_sensor_uuid, None)
    #print(observations[])
    return observations

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


def collate_fn(batch):
    """Each sample in batch: (
            obs,
            prev_actions,
            oracle_actions,
            inflec_weight,
        )
    """

    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(pad_amount, *t.size()[1:])
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))

    observations_batch = list(transposed[0])
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])
    B = len(prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(observations_batch[bid][sensor])

    observations_batch = new_observations_batch

    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid], max_traj_len, fill_val=1.0
            )

        prev_actions_batch[bid] = _pad_helper(prev_actions_batch[bid], max_traj_len)
        corrected_actions_batch[bid] = _pad_helper(
            corrected_actions_batch[bid], max_traj_len
        )
        weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(observations_batch[sensor], dim=1)
        observations_batch[sensor] = observations_batch[sensor].view(
            -1, *observations_batch[sensor].size()[2:]
        )

    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(corrected_actions_batch, dtype=torch.float)
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch,
        prev_actions_batch.view(-1, 1),
        not_done_masks.view(-1, 1),
        corrected_actions_batch,
        weights_batch,
    )


def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]


class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw,
        inflection_weight_coef=1.0,
        lmdb_map_size=1e9,
        batch_size=1,
    ):
        super().__init__()
        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 100
        self._preload = []
        self.batch_size = batch_size

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
        ) as lmdb_env:
            self.length = lmdb_env.stat()["entries"]

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for _ in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    new_preload.append(
                        msgpack_numpy.unpackb(
                            txn.get(str(self.load_ordering.pop()).encode()), raw=False
                        )
                    )

                    lengths.append(len(new_preload[-1][0]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

        return self._preload.pop()

    def __next__(self):
        obs, prev_actions, oracle_actions = self._load_next()

        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (obs, prev_actions, oracle_actions, self.inflec_weights[inflections])

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = 0
            end = self.length
        else:
            per_worker = int(np.ceil(self.length / worker_info.num_workers))

            start = per_worker * worker_info.id
            end = min(start + per_worker, self.length)

        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(_block_shuffle(list(range(start, end)), self.preload_size))
        )

        return self


@baseline_registry.register_trainer(name="speaker_dagger")
class SpeakerDaggerTrainer(BaseRLTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.envs = None

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available() and not config.USE_CPU
            else torch.device("cpu")
        )
        self.lmdb_features_dir = self.config.DAGGER.LMDB_FEATURES_DIR.format(
            split=config.TASK_CONFIG.DATASET.SPLIT
        )

    def _setup_actor_critic_agent(
        self, config: Config, load_from_ckpt: bool, ckpt_path: str
    ) -> None:
        r"""Sets up actor critic and agent.
        Args:
            config: MODEL config
        Returns:
            None
        """
        config.defrost()
        config.TORCH_GPU_ID = self.config.TORCH_GPU_ID
        config.USE_CPU = self.config.USE_CPU
        config.freeze()

        if config.LXMERT.use_seq2seq:
            self.actor_critic = GlobalLXMERTSeq2seqPolicy(
                observation_space=self.envs.observation_spaces[0],
                action_space=self.envs.action_spaces[0],
                model_config=config,
            )
        elif config.LXMERT.use_cma:
            self.actor_critic = GlobalLXMERTCMAPolicy(
                observation_space=self.envs.observation_spaces[0],
                action_space=self.envs.action_spaces[0],
                model_config=config,
            )
        else:
            raise NotImplementedError
        
        self.actor_critic.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), 
            lr=self.config.DAGGER.LR
        )
        if load_from_ckpt:
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            self.actor_critic.load_state_dict(ckpt_dict["state_dict"])
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}")
        logger.info("Finished setting up actor critic model.")

    def save_checkpoint(self, file_name) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.actor_critic.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name))

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def _update_dataset(self, data_it):
        if torch.cuda.is_available() and not self.config.USE_CPU:
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

        if self.envs is None:
            self.envs = construct_envs(
                self.config, 
                get_env_class(self.config.ENV_NAME)
            )

        recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device)
        
        observations = self.envs.reset()
        observations = transform_obs(
            observations, 
            self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(
            observations, self.device, 
            skip_list=[
                'template_sensor', 
                'semantic', 
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
            ]
        )

        episodes = [[] for _ in range(self.envs.num_envs)]
        skips = [False for _ in range(self.envs.num_envs)]
        # Populate dones with False initially
        dones = [False for _ in range(self.envs.num_envs)]

        # https://arxiv.org/pdf/1011.0686.pdf
        # Theoretically, any beta function is fine so long as it converges to
        # zero as data_it -> inf. The paper suggests starting with beta = 1 and
        # exponential decay.
        if self.config.DAGGER.P == 0.0:
            # in Python 0.0 ** 0.0 == 1.0, but we want 0.0
            beta = 0.0
        else:
            beta = self.config.DAGGER.P ** data_it

        ensure_unique_episodes = beta == 1.0

        def hook_builder(tgt_tensor):
            def hook(m, i, o):
                tgt_tensor.set_(o.to(torch.device("cpu")))
            return hook

        def frcnn_hook_builder(tgt_dict):
            def hook(m, i, o):
                for key in o:
                    tgt_dict[key] = torch.zeros(
                        (1,), device="cpu",dtype=o[key].dtype
                    )
                    tgt_dict[key].set_(o[key].to(torch.device("cpu")))
            return hook

        def proposal_hook_builder(tgt_list):
            def hook(m, i, o):
                boxes, logits = o
                for idx in range(len(boxes)):
                    tgt_list.append(torch.zeros(
                        (1,), device="cpu",dtype=boxes[idx].dtype
                    ))
                    tgt_list[idx].set_(boxes[idx].to(torch.device("cpu")))
            return hook

        def lxmert_hook_builder(tgt_vis, tgt_lang, tgt_dummy):
            def hook(m, i, o):
                # note: this follows the forward function in modeling_lxmert.py
                # o is the encoder output
                vis_enc_out, lng_enc_out = o[:2]
                
                vis_out = vis_enc_out[0][-1]
                tgt_vis.set_(vis_out.to(torch.device("cpu")))
                
                lang_out = lng_enc_out[0][-1]
                tgt_lang.set_(lang_out.to(torch.device("cpu")))
                
                # moving everything else to cpu. 
                # the code below is super inneficient. need to find a better
                # way of handling this
                u = 0
                for j in range(len(vis_enc_out)):
                    if vis_enc_out[j] == None:
                        continue
                    
                    for k in range(len(vis_enc_out[j])):
                        tgt_dummy.append(torch.zeros((1,), device="cpu",))
                        tgt_dummy[u].set_(vis_enc_out[j][k].to(torch.device("cpu")))
                        u += 1
            
                for j in range(len(lng_enc_out)):
                    if lng_enc_out[j] == None:
                        continue
                    
                    for k in range(len(lng_enc_out[j])):
                        tgt_dummy.append(torch.zeros((1,), device="cpu",))
                        tgt_dummy[u].set_(lng_enc_out[j][k].to(torch.device("cpu")))
                        u += 1
                        
            return hook
        
        depth_features = None
        depth_hook = None
        if self.config.MODEL.DEPTH_ENCODER.cnn_type == "VlnResnetDepthEncoder":
            depth_features = torch.zeros((1,), device="cpu")
            depth_hook = self.actor_critic.net.depth_encoder.visual_encoder.register_forward_hook(
                hook_builder(depth_features)
            )
        
        # not sure if we need these hooks
        frcnn_features = {}
        frcnn_hook = self.actor_critic.net.lxmert_encoder.frcnn.backbone.register_forward_hook(
            frcnn_hook_builder(frcnn_features)
        )
        
        box_features = []
        box_hook = self.actor_critic.net.lxmert_encoder.frcnn.proposal_generator.register_forward_hook(
            proposal_hook_builder(box_features)
        )
    
        lxmert_dummy = []
        lxmert_vis_features = torch.zeros((1,), device="cpu")
        lxmert_lang_features = torch.zeros((1,), device="cpu")
        lxmert_features_hook = self.actor_critic.net.lxmert_encoder.lxmert.encoder.register_forward_hook(
            lxmert_hook_builder(
                lxmert_vis_features, lxmert_lang_features, lxmert_dummy
            )
        )
        
        lxmert_pooled_lang_features = torch.zeros((1,), device="cpu")
        lxmert_pooled_lang_hook = self.actor_critic.net.lxmert_encoder.lxmert.pooler.register_forward_hook(
            hook_builder(lxmert_pooled_lang_features)
        )
        
        collected_eps = 0
        ep_ids_collected = None
        if ensure_unique_episodes:
            ep_ids_collected = set(
                [ep.episode_id for ep in self.envs.current_episodes()]
            )
        
        # gets the ids of the previously saved trajectories
        # skip_episodes = None
        # if self.config.DAGGER.LMDB_RESUME:
        #     collected_file = open(self.config.DAGGER.LMDB_PROGRESS_FILE, 'r')
        #     skip_episodes = [int(e) for e in collected_file.read().splitlines()]
        #     for episode in skip_episodes:
        #         ep_ids_collected.add(episode)
        lmdb_env = lmdb.open(
            self.lmdb_features_dir, 
            map_size=int(self.config.DAGGER.LMDB_MAP_SIZE)
        )
        update_size = self.config.DAGGER.UPDATE_SIZE
        
        with tqdm.tqdm(total=update_size) as pbar, torch.no_grad():
            start_id = lmdb_env.stat()["entries"]
            # pbar.update(start_id)
            txn = lmdb_env.begin(write=True)
            # collected_eps = start_id

            while collected_eps < update_size:
                current_episodes = None
                envs_to_pause = None
                if ensure_unique_episodes:
                    envs_to_pause = []
                    current_episodes = self.envs.current_episodes()
                
                # if not skip_episodes is None:
                #     for i in range(self.envs.num_envs):
                #         curr_episode_id = current_episodes[i].episode_id
                #         if curr_episode_id in skip_episodes:
                #             logger.info(f"Skipping episode: {curr_episode_id}")
                #             self.envs.step([0])
                #     current_episodes = self.envs.current_episodes()

                for i in range(self.envs.num_envs):
                    if dones[i] and not skips[i]:
                        ep = episodes[i]
                        traj_obs = batch_obs(
                            [step[0] for step in ep], device=torch.device("cpu"),
                            skip_list=[
                                'template_sensor', 
                                'semantic',
                                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
                            ]
                        )
                        del traj_obs["vln_oracle_action_sensor"]
                        for k, v in traj_obs.items():
                            traj_obs[k] = v.numpy()

                        transposed_ep = [
                            traj_obs,
                            np.array([step[1] for step in ep], dtype=np.int64),
                            np.array([step[2] for step in ep], dtype=np.int64),
                        ]
                        # print(transposed_ep)
                        txn.put(
                            str(start_id + collected_eps).encode(),
                            msgpack_numpy.packb(transposed_ep, use_bin_type=True),
                        )

                        pbar.update()
                        collected_eps += 1

                        if (
                            collected_eps % self.config.DAGGER.LMDB_COMMIT_FREQUENCY
                        ) == 0:
                            txn.commit()
                            txn = lmdb_env.begin(write=True)
                            
                            # write progress file
                            # eps_ids_collected_file = open(
                            #     self.config.DAGGER.LMDB_PROGRESS_FILE, 'w'
                            # )
                            # eps_list = list(ep_ids_collected)
                            # for eps in eps_list:
                            #     eps_ids_collected_file.write(f"{eps}\n")
                            # eps_ids_collected_file.close()

                        if ensure_unique_episodes:
                            if current_episodes[i].episode_id in ep_ids_collected:
                                envs_to_pause.append(i)
                            else:
                                ep_ids_collected.add(current_episodes[i].episode_id)

                    if dones[i]:
                        episodes[i] = []

                if ensure_unique_episodes:
                    (
                        self.envs,
                        recurrent_hidden_states,
                        not_done_masks,
                        prev_actions,
                        batch,
                    ) = self._pause_envs(
                        envs_to_pause,
                        self.envs,
                        recurrent_hidden_states,
                        not_done_masks,
                        prev_actions,
                        batch,
                    )
                    if self.envs.num_envs == 0:
                        break

                (_, actions, _, recurrent_hidden_states) = self.actor_critic.act(
                    batch,
                    recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                actions = torch.where(
                    torch.rand_like(actions, dtype=torch.float) < beta,
                    batch["vln_oracle_action_sensor"].long(),
                    actions,
                )

                for i in range(self.envs.num_envs):
                    
                    if lxmert_vis_features is not None:
                        # shape (bboxes, 768)
                        observations[i]["lxmert_vision"] = lxmert_vis_features[i]
                        del observations[i]["rgb"]
                    
                    if lxmert_lang_features is not None:
                        
                        if self.config.MODEL.LXMERT.use_pooled:
                            # shape (1, 768)
                            observations[i]["lxmert_lang"] = lxmert_pooled_lang_features[i]
                        else:
                            # shape (tokens, 768)
                            observations[i]["lxmert_lang"] = lxmert_lang_features[i]
                        del observations[i]["lxmertinstruction"]
                    
                    if depth_features is not None:
                        # shape (128, 4, 4)
                        observations[i]["depth_features"] = depth_features[i]
                        del observations[i]["depth"]
                    
                    episodes[i].append(
                        (
                            observations[i],
                            prev_actions[i].item(),
                            batch["vln_oracle_action_sensor"][i].item(),
                        )
                    )
                
                skips = batch["vln_oracle_action_sensor"].long() == -1
                actions = torch.where(skips, torch.zeros_like(actions), actions)
                skips = skips.squeeze(-1).to(device="cpu", non_blocking=True)

                prev_actions.copy_(actions)

                outputs = self.envs.step([a[0].item() for a in actions])
                observations, rewards, dones, _ = [list(x) for x in zip(*outputs)]

                not_done_masks = torch.tensor(
                    [[0.0] if done else [1.0] for done in dones],
                    dtype=torch.float,
                    device=self.device,
                )

                observations = transform_obs(
                    observations, 
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
                )
                batch = batch_obs(
                    observations, 
                    self.device, 
                    skip_list=[
                        'template_sensor', 
                        'semantic',
                        self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
                    ]
                )

            txn.commit()

        self.envs.close()
        self.envs = None
        
        if depth_hook is not None:
            depth_hook.remove()

        if frcnn_hook is not None:
            frcnn_hook.remove()
        
        if box_hook is not None:
            box_hook.remove()
            
        if lxmert_features_hook is not None:
            lxmert_features_hook.remove()
            
        if lxmert_pooled_lang_hook is not None:
            lxmert_pooled_lang_hook.remove()
            
            
    def _update_agent(
        self, observations, prev_actions, not_done_masks, corrected_actions, weights
    ):
        T, N = corrected_actions.size()
        self.optimizer.zero_grad()

        recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            N,
            self.config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )

        AuxLosses.clear()

        distribution = self.actor_critic.build_distribution(
            observations, recurrent_hidden_states, prev_actions, not_done_masks
        )

        logits = distribution.logits
        logits = logits.view(T, N, -1)

        action_loss = F.cross_entropy(
            logits.permute(0, 2, 1), corrected_actions, reduction="none"
        )
        action_loss = ((weights * action_loss).sum(0) / weights.sum(0)).mean()

        aux_mask = (weights > 0).view(-1)
        aux_loss = AuxLosses.reduce(aux_mask)

        loss = action_loss + aux_loss
        loss.backward()

        self.optimizer.step()

        if isinstance(aux_loss, torch.Tensor):
            return loss.item(), action_loss.item(), aux_loss.item()
        else:
            return loss.item(), action_loss.item(), aux_loss

    def train(self) -> None:
        r"""Main method for training DAgger.

        Returns:
            None
        """
        os.makedirs(self.lmdb_features_dir, exist_ok=True)
        os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
        
        if self.config.DAGGER.PRELOAD_LMDB_FEATURES:
            try:
                lmdb.open(self.lmdb_features_dir, readonly=True)
            except lmdb.Error as err:
                logger.error("Cannot open database for teacher forcing preload.")
                raise err
        else:
            with lmdb.open(
                self.lmdb_features_dir, map_size=int(self.config.DAGGER.LMDB_MAP_SIZE)
            ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                txn.drop(lmdb_env.open_db())
                
        # if self.config.DAGGER.PRELOAD_LMDB_FEATURES:
        #     try:
        #         lmdb.open(self.lmdb_features_dir, readonly=True)
        #     except lmdb.Error as err:
        #         logger.error("Cannot open database for teacher forcing preload.")
        #         raise err
        # elif not self.config.DAGGER.LMDB_RESUME:
        #     # this will delete the db
        #     with lmdb.open(
        #         self.lmdb_features_dir, map_size=int(self.config.DAGGER.LMDB_MAP_SIZE)
        #     ) as lmdb_env, lmdb_env.begin(write=True) as txn:
        #         txn.drop(lmdb_env.open_db())
        #     file = open(self.config.DAGGER.LMDB_PROGRESS_FILE, 'w')
        #     file.close()
        #     # otherwise, the db will be kept to resume the trajectory generation
        #     # process

        split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = split

        # if doing teacher forcing, don't switch the scene until it is complete
        if self.config.DAGGER.P == 1.0:
            self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
                -1
            )
        self.config.freeze()

        if self.config.DAGGER.PRELOAD_LMDB_FEATURES:
            # when preloadeding features, its quicker to just load one env as we just
            # need the observation space from it.
            single_proc_config = self.config.clone()
            single_proc_config.defrost()
            single_proc_config.NUM_PROCESSES = 1
            single_proc_config.freeze()
            self.envs = construct_envs(
                single_proc_config, 
                get_env_class(self.config.ENV_NAME)
            )
        else:
            self.envs = construct_envs(
                self.config, 
                get_env_class(self.config.ENV_NAME)
            )

        self._setup_actor_critic_agent(
            self.config.MODEL,
            self.config.DAGGER.LOAD_FROM_CKPT,
            self.config.DAGGER.CKPT_TO_LOAD,
        )

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.actor_critic.parameters())
            )
        )
        logger.info(
            "agent number of trainable parameters: {}".format(
                sum(
                    p.numel() for p in self.actor_critic.parameters() if p.requires_grad
                )
            )
        )

        if self.config.DAGGER.PRELOAD_LMDB_FEATURES:
            self.envs.close()
            del self.envs
            self.envs = None

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs, purge_step=0
        ) as writer:
            for dagger_it in range(self.config.DAGGER.ITERATIONS):
                step_id = 0
                if not self.config.DAGGER.PRELOAD_LMDB_FEATURES:
                    self._update_dataset(
                        dagger_it + (1 if self.config.DAGGER.LOAD_FROM_CKPT else 0)
                    )

                if torch.cuda.is_available() and not self.config.USE_CPU:
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                gc.collect()

                dataset = IWTrajectoryDataset(
                    self.lmdb_features_dir,
                    self.config.DAGGER.USE_IW,
                    inflection_weight_coef=self.config.MODEL.inflection_weight_coef,
                    lmdb_map_size=self.config.DAGGER.LMDB_MAP_SIZE,
                    batch_size=self.config.DAGGER.BATCH_SIZE,
                )
                diter = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.config.DAGGER.BATCH_SIZE,
                    shuffle=False,
                    collate_fn=collate_fn,
                    pin_memory=False,
                    drop_last=True,  # drop last batch if smaller
                    num_workers=0,
                    worker_init_fn=set_worker_sharing_strategy
                )

                AuxLosses.activate()
                for epoch in tqdm.trange(self.config.DAGGER.EPOCHS):
                    for batch in tqdm.tqdm(
                        diter, total=dataset.length // dataset.batch_size, leave=False
                    ):
                        (
                            observations_batch,
                            prev_actions_batch,
                            not_done_masks,
                            corrected_actions_batch,
                            weights_batch,
                        ) = batch
                        
                        observations_batch = {
                            k: v.to(device=self.device, non_blocking=True)
                            for k, v in observations_batch.items()
                        }

                        try:
                            loss, action_loss, aux_loss = self._update_agent(
                                observations_batch,
                                prev_actions_batch.to(
                                    device=self.device, non_blocking=True
                                ),
                                not_done_masks.to(
                                    device=self.device, non_blocking=True
                                ),
                                corrected_actions_batch.to(
                                    device=self.device, non_blocking=True
                                ),
                                weights_batch.to(device=self.device, non_blocking=True),
                            )
                        except:
                            logger.info(
                                "ERROR: failed to update agent. Updating agent with batch size of 1."
                            )
                            loss, action_loss, aux_loss = 0, 0, 0
                            prev_actions_batch = prev_actions_batch.cpu()
                            not_done_masks = not_done_masks.cpu()
                            corrected_actions_batch = corrected_actions_batch.cpu()
                            weights_batch = weights_batch.cpu()
                            observations_batch = {
                                k: v.cpu() for k, v in observations_batch.items()
                            }
                            for i in range(not_done_masks.size(0)):
                                output = self._update_agent(
                                    {
                                        k: v[i].to(
                                            device=self.device, non_blocking=True
                                        )
                                        for k, v in observations_batch.items()
                                    },
                                    prev_actions_batch[i].to(
                                        device=self.device, non_blocking=True
                                    ),
                                    not_done_masks[i].to(
                                        device=self.device, non_blocking=True
                                    ),
                                    corrected_actions_batch[i].to(
                                        device=self.device, non_blocking=True
                                    ),
                                    weights_batch[i].to(
                                        device=self.device, non_blocking=True
                                    ),
                                )
                                loss += output[0]
                                action_loss += output[1]
                                aux_loss += output[2]

                        logger.info(f"train_loss: {loss}")
                        logger.info(f"train_action_loss: {action_loss}")
                        logger.info(f"train_aux_loss: {aux_loss}")
                        logger.info(f"Batches processed: {step_id}.")
                        logger.info(f"On DAgger iter {dagger_it}, Epoch {epoch}.")
                        writer.add_scalar(f"train_loss_iter_{dagger_it}", loss, step_id)
                        writer.add_scalar(
                            f"train_action_loss_iter_{dagger_it}", action_loss, step_id
                        )
                        writer.add_scalar(
                            f"train_aux_loss_iter_{dagger_it}", aux_loss, step_id
                        )
                        step_id += 1

                    self.save_checkpoint(
                        f"ckpt.{dagger_it * self.config.DAGGER.EPOCHS + epoch}.pth"
                    )
                AuxLosses.deactivate()

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        recurrent_hidden_states,
        not_done_masks,
        prev_actions,
        batch,
    ):
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            recurrent_hidden_states = recurrent_hidden_states[:, state_index]
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

        return (envs, recurrent_hidden_states, not_done_masks, prev_actions, batch)

    def _eval_checkpoint(
        self, checkpoint_path: str, writer: TensorboardWriter, checkpoint_index: int = 0
    ) -> None:
        r"""Evaluates a single checkpoint. Assumes episode IDs are unique.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        
        # TODO: create a dictionary where you'll save the episodes.
        
        # checkpoint_path = self.config.EVAL.CHECKPOINT_FILE
        logger.info(f"Running checkpoint_path: {checkpoint_path}")
        checkpoint_index = int(checkpoint_path.split(".")[1])
        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(
                self.load_checkpoint(checkpoint_path, map_location="cpu")["config"]
            )
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.TASK.NDTW.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.TASK.SDTW.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        if len(config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")

        config.freeze()

        # setup agent
        self.envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )
        self.device = (
            torch.device("cuda", config.TORCH_GPU_ID)
            if torch.cuda.is_available() and not config.USE_CPU
            else torch.device("cpu")
        )

        self._setup_actor_critic_agent(config.MODEL, True, checkpoint_path)

        observations = self.envs.reset()
        observations = transform_obs(
            observations, config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(
            observations, self.device, 
            skip_list=[
                'template_sensor', 
                'semantic', 
                config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
            ]
        )

        eval_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            config.NUM_PROCESSES,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(config.NUM_PROCESSES, 1, device=self.device)

        stats_episodes = {}  # dict of dicts that stores stats per episode

        rgb_frames = None
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)
            rgb_frames = [[] for _ in range(config.NUM_PROCESSES)]
        pbar = tqdm.tqdm(total=config.EVAL.EPISODE_COUNT) 
        self.actor_critic.eval()
        while (
            self.envs.num_envs > 0 and len(stats_episodes) < config.EVAL.EPISODE_COUNT
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (_, actions, _, eval_recurrent_hidden_states) = self.actor_critic.act(
                    batch,
                    eval_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                )
                prev_actions.copy_(actions)

            # TODO: if you're using the VLNCEDaggerSpeakerEnv, check that 
            # the infos field contains the trajectory, and that it is being 
            # updated with every step. And that it's being reset with every episode
            outputs = self.envs.step([a[0].item() for a in actions])
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            # reset envs and observations if necessary
            for i in range(self.envs.num_envs):
                if len(config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(
                        frame, current_episodes[i].instruction.instruction_text
                    )
                    rgb_frames[i].append(frame)

                if not dones[i]:
                    continue

                stats_episodes[current_episodes[i].episode_id] = infos[i]
                observations[i] = self.envs.reset_at(i)[0]
                prev_actions[i] = torch.zeros(1, dtype=torch.long)
                pbar.update(1)	
                if len(config.VIDEO_OPTION) > 0:
                    # TODO: here the episode i is done, so get episode info
                    # Episode info is in current_episodes[i]
                    
                    # TODO: append the trajectory info to the episode
                    
                    # NOTE: episode info should be appended in the same format 
                    # the datasets. example:
                    #  "episodes": [
                    # {
                    #   "episode_id": 1,
                    #   "trajectory_id": 15,
                    #   "scene_id": "mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb",
                    #   "start_position": [
                    #     15.068599700927734, 0.17162801325321198, -4.4848198890686035
                    #   ],
                    #   "start_rotation": [0, 0.6198966754439885, 0, -0.7846834468583432],
                    #   "info": { "geodesic_distance": 7.9608235359191895 },
                    #   "goals": [
                    #     {
                    #       "position": [
                    #         13.04640007019043, 0.17162801325321198, 1.8739700317382812
                    #       ],
                    #       "radius": 3.0
                    #     }
                    #   ],
                    #   "instruction": {
                    #     "instruction_text": "Exit the bedroom and turn left. Walk straight passing the gray couch and stop near the rug. ",
                    #     "instruction_tokens": [
                    #       816, 2202, 246, 103, 2300, 1251, 9, 2384, 2112, 1588, 2202, 1009, 549,
                    #       103, 2104, 1437, 2202, 1856, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    #       0, 0, 0, 0, 0, 0, 0
                    #     ]
                    #   },
                    #   "reference_path": [
                    #     [15.068599700927734, 0.17162801325321198, -4.4848198890686035],
                    #     [13.649399757385254, 0.17162801325321198, -4.241700172424316],
                    #     [12.57610034942627, 0.17162801325321198, -4.270150184631348],
                    #     [12.461600303649902, 0.17162801325321198, -2.3902199268341064],
                    #     [12.857000350952148, 0.17162801325321198, -0.06946899741888046],
                    #     [13.04640007019043, 0.17162801325321198, 1.8739700317382812]
                    #   ]
                    # },
                    #  ...
                    # ]
                    # THE ONLY ADDITIONAL INFO WOULD BE THE TRAJECTORY INFO
                    
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=current_episodes[i].episode_id,
                        checkpoint_idx=checkpoint_index,
                        metrics={
                            "spl": stats_episodes[current_episodes[i].episode_id]["spl"]
                        },
                        tb_writer=writer,
                    )

                    del stats_episodes[current_episodes[i].episode_id]["top_down_map"]
                    del stats_episodes[current_episodes[i].episode_id]["collisions"]
                    rgb_frames[i] = []

            observations = transform_obs(
                observations, 
                config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
            )
            batch = batch_obs(
                observations, self.device, 
                skip_list=[
                    'template_sensor', 
                    'semantic', 
                    config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
                ]
            )

            envs_to_pause = []
            next_episodes = self.envs.current_episodes()

            for i in range(self.envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            (
                self.envs,
                eval_recurrent_hidden_states,
                not_done_masks,
                prev_actions,
                batch,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                eval_recurrent_hidden_states,
                not_done_masks,
                prev_actions,
                batch,
            )
        pbar.close()
        self.envs.close()
        
        # TODO: all episodes are done at this point. So save the dictionary of
        # episodes 
        #print(stats_episodes.values())
        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()]) / num_episodes
            )
        aggregated_stats['successful_steps'] = ( sum([v['steps_taken'] for v in stats_episodes.values() if v['success'] == 1.0])/ (aggregated_stats['success']*num_episodes))   
        
        split = config.TASK_CONFIG.DATASET.SPLIT
        with open(f"stats_ckpt_{checkpoint_index}_{split}.json", "w") as f:
            json.dump(aggregated_stats, f, indent=4)

        logger.info(f"Episodes evaluated: {num_episodes}")
        checkpoint_num = checkpoint_index + 1
        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.6f}")
            writer.add_scalar(f"eval_{split}_{k}", v, checkpoint_num)

    def inference(self) -> None:
        r"""Runs inference on a single checkpoint, creating a path predictions file."""

        checkpoint_path = self.config.INFERENCE.CKPT_PATH 
        logger.info(f"checkpoint_path: {checkpoint_path}")

        if self.config.INFERENCE.USE_CKPT_CONFIG:
            config = self._setup_eval_config(
                self.load_checkpoint(checkpoint_path, map_location="cpu")["config"]
            )
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        config.TASK_CONFIG.TASK.MEASUREMENTS = []
        config.TASK_CONFIG.TASK.SENSORS = ["INSTRUCTION_SENSOR"]
        config.ENV_NAME = "VLNCEInferenceEnv"
        config.freeze()

        # setup agent
        self.envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )
        self.device = (
            torch.device("cuda", config.TORCH_GPU_ID)
            if torch.cuda.is_available() and not config.USE_CPU
            else torch.device("cpu")
        )

        self._setup_actor_critic_agent(config.MODEL, True, checkpoint_path)
        self.actor_critic.eval()

        observations = self.envs.reset()
        observations = transform_obs(
            observations, config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,inference = True
        )
        batch = batch_obs(
            observations, self.device, 
            skip_list=[
                'template_sensor', 
                'semantic', 
                config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
            ]
        )

        rnn_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            config.NUM_PROCESSES,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(config.NUM_PROCESSES, 1, device=self.device)

        episode_predictions = defaultdict(list)

        # populate episode_predictions with the starting state
        current_episodes = self.envs.current_episodes()
        for i in range(self.envs.num_envs):
            episode_predictions[current_episodes[i].episode_id].append(
                self.envs.call_at(i, "get_info", {"observations": {}})
            )

        with tqdm.tqdm(
            total=sum(self.envs.count_episodes()),
            desc=f"[inference:{self.config.INFERENCE.SPLIT}]",
        ) as pbar:
            while self.envs.num_envs > 0:
                current_episodes = self.envs.current_episodes()

                with torch.no_grad():
                    (_, actions, _, rnn_states) = self.actor_critic.act(
                        batch,
                        rnn_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=True,
                    )
                    prev_actions.copy_(actions)

                outputs = self.envs.step([a[0].item() for a in actions])
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]

                not_done_masks = torch.tensor(
                    [[0.0] if done else [1.0] for done in dones],
                    dtype=torch.float,
                    device=self.device,
                )

                # reset envs and observations if necessary
                for i in range(self.envs.num_envs):
                    episode_predictions[current_episodes[i].episode_id].append(infos[i])
                    if not dones[i]:
                        continue

                    observations[i] = self.envs.reset_at(i)[0]
                    prev_actions[i] = torch.zeros(1, dtype=torch.long)
                    pbar.update(1)

                observations = transform_obs(
                    observations, config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID, inference=True
                )
                batch = batch_obs(
                    observations, self.device, 
                    skip_list=[
                        'template_sensor', 
                        'semantic', 
                        config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
                    ]
                )

                envs_to_pause = []
                next_episodes = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue

                    if next_episodes[i].episode_id in episode_predictions:
                        envs_to_pause.append(i)
                    else:
                        episode_predictions[next_episodes[i].episode_id].append(
                            self.envs.call_at(i, "get_info", {"observations": {}})
                        )

                (
                    self.envs,
                    rnn_states,
                    not_done_masks,
                    prev_actions,
                    batch,
                ) = self._pause_envs(
                    envs_to_pause,
                    self.envs,
                    rnn_states,
                    not_done_masks,
                    prev_actions,
                    batch,
                )

        self.envs.close()

        with open(config.INFERENCE.PREDICTIONS_FILE, "w") as f:
            json.dump(episode_predictions, f, indent=2)

        logger.info(f"Predictions saved to: {config.INFERENCE.PREDICTIONS_FILE}")
