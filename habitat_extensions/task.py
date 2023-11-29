import attr
import gzip
import json
import os

import numpy as np

from typing import List, Optional, Tuple, Type, Any
from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.core.utils import (
    DatasetFloatJSONEncoder,
    not_none_validator
)
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import (
    NavigationGoal,
    NavigationTask,
    # merge_sim_episode_config,
)
from habitat.tasks.nav.nav import NavigationTask
from habitat.tasks.vln.vln import InstructionData, VLNEpisode
from gym import spaces

ALL_SCENES_MASK = "*"
CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_dataset/"

# TODO: add an optional field for the trajectories
@attr.s(auto_attribs=True, kw_only=True)
class VLNExtendedEpisode(VLNEpisode):
    r"""
    instruction_index_string: optional identifier of instruction.
    """
    instruction_index_string: str = attr.ib(default=None)
    goals: Optional[List[NavigationGoal]] = attr.ib(default=None)
    reference_path: Optional[List[List[float]]] = attr.ib(default=None)
    objects : Optional[List[Tuple]] = attr.ib(default=None)

@registry.register_dataset(name="VLN-CE-v1")
class VLNCEDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads a Vision and Language
    Navigation dataset.
    """

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def _scene_from_episode(episode: VLNExtendedEpisode) -> str:
        r"""Helper method to get the scene name from an episode.  Assumes
        the scene_id is formated /path/to/<scene_name>.<ext>
        """
        return os.path.splitext(os.path.basename(episode.scene_id))[0]

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        r"""Return a sorted list of scenes
        """
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        scenes = {cls._scene_from_episode(episode)
                  for episode in dataset.episodes}

        return sorted(list(scenes))

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return

        dataset_filename = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        if ALL_SCENES_MASK not in config.CONTENT_SCENES:
            scenes_to_load = set(config.CONTENT_SCENES)
            self.episodes = [
                episode
                for episode in self.episodes
                if self._scene_from_episode(episode) in scenes_to_load
            ]

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:

        deserialized = json.loads(json_str)
        self.instruction_vocab = VocabDict(
            word_list=deserialized["instruction_vocab"]["word_list"]
        )

        for episode in deserialized["episodes"]:
            episode = VLNExtendedEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX):
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = InstructionData(**episode.instruction)
            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)


def merge_sim_episode_with_heading_config(
        sim_config: Config, episode: Type[Episode]) -> Any:
    sim_config = merge_sim_episode_config(sim_config, episode)
    # sim_config.defrost()
    # sim_cfg.objects = [episode.objects.__dict__]
    return sim_config


#
# Waypoint dataset
# ------------------------------------------------------------------------------

@registry.register_task(name="LocalNavigatorTask-v0")
class NavigatorTask(NavigationTask):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def overwrite_sim_config(self, sim_config, episode):
        return merge_sim_episode_with_heading_config(sim_config, episode)
    
    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)


@registry.register_task(name="WaypointTask-v0")
class WaypointTask(NavigationTask):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def overwrite_sim_config(self, sim_config, episode):
        return merge_sim_episode_with_heading_config(sim_config, episode)
    
    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)


@attr.s(auto_attribs=True, kw_only=True)
class Waypoint:
    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: float = attr.ib(default=None, validator=not_none_validator)
    rel_distance: float = attr.ib(default=None, validator=not_none_validator)
    rel_heading: float = attr.ib(default=None, validator=not_none_validator)
    rel_elevation: float = attr.ib(default=None, validator=not_none_validator)


@attr.s(auto_attribs=True, kw_only=True)
class WaypointEpisode(Episode):
    path_id: int = attr.ib(default=None, validator=not_none_validator)
    is_goal: bool = attr.ib(default=None, validator=not_none_validator)
    num_waypoint: int =  attr.ib(default=None, validator=not_none_validator)
    # instr_text: str = attr.ib(default=None, validator=not_none_validator)
    # instr_tokens: List[int] = attr.ib(default=None, validator=not_none_validator)
    goals: List[Waypoint] = attr.ib(default=None, validator=not_none_validator)


@registry.register_dataset(name="Waypoint-v0")
class WaypointDatasetv0(Dataset):
    episodes: List[WaypointEpisode]
    
    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return

        datasetfile_path = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        deserialized = json.loads(json_str)

        for episode in deserialized["episodes"]:
            episode = WaypointEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = (
                        episode.scene_id[
                            len(DEFAULT_SCENE_PATH_PREFIX):
                        ]
                    )

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            if episode.goals is not None:
                for g, goal in enumerate(episode.goals):
                    episode.goals[g] = Waypoint(**goal)

            self.episodes.append(episode)

#
# Waypoint dataset
# ------------------------------------------------------------------------------

@attr.s(auto_attribs=True, kw_only=True)
class GPlannerEpisode(Episode):
    path_id: int = attr.ib(default=None, validator=not_none_validator)
    goals: List[NavigationGoal] = attr.ib(default=None, validator=not_none_validator)
    path: List[int] = attr.ib(default=None, validator=not_none_validator)
    # instruction: str = attr.ib(default=None, validator=not_none_validator)

@registry.register_dataset(name="GPlannerDataset-v0")
class GPlannerDatasetv0(Dataset):
    episodes: List[GPlannerEpisode]
    
    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return

        datasetfile_path = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        deserialized = json.loads(json_str)

        for episode in deserialized["episodes"]:
            episode = GPlannerEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = (
                        episode.scene_id[
                            len(DEFAULT_SCENE_PATH_PREFIX):
                        ]
                    )

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            if episode.goals is not None:
                for g, goal in enumerate(episode.goals):
                    episode.goals[g] = NavigationGoal(**goal)

            self.episodes.append(episode)
            
def merge_sim_episode_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config.defrost()
    # here's where the scene update happens, extract the scene name out of the path
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if (
        episode.start_position is not None
        and episode.start_rotation is not None
    ):
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation
        agent_cfg.GOAL_POSITION = episode.goals[0].position
        # agent_cfg.SOUND_ID = episode.info['sound'] + '.wav'
        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config

@registry.register_task_action
class TeleportAction(SimulatorTaskAction):
    # TODO @maksymets: Propagate through Simulator class
    COORDINATE_EPSILON = 1e-6
    COORDINATE_MIN = -62.3241 - COORDINATE_EPSILON
    COORDINATE_MAX = 90.0399 + COORDINATE_EPSILON

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "TELEPORT"

    def step(
        self,
        *args: Any,
        position: List[float],
        rotation: List[float],
        **kwargs: Any,
    ):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """

        if not isinstance(rotation, list):
            rotation = list(rotation)

        if not self._sim.is_navigable(position):
            print("Not navigable")
            return self._sim.get_observations_at()

        return self._sim.get_observations_at(
            position=position, rotation=rotation, keep_agent_at_new_pose=True
        )

    @property
    def action_space(self):
        return spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(3,), dtype=np.float32
                    ),
                "rotation": spaces.Box(
                    low=np.array([-1.0, -1.0, -1.0, -1.0]),
                    high=np.array([1.0, 1.0, 1.0, 1.0]),
                    dtype=np.float32,
                ),
            }
        )
