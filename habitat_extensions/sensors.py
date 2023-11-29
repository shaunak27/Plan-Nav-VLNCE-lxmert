import gzip
import json
import sys
sys.path.append('..')
from utils.region import run_exp
import torch
import cv2
import numpy as np
import time
import random
from operator import itemgetter
from gym import spaces
from habitat import config
from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes, Simulator, Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.tasks.vln.vln import VLNEpisode
from habitat.utils.geometry_utils import quaternion_to_list
from PIL import Image
from transformers import LxmertTokenizer
from typing import Any, Union, Type, Union, Dict

from habitat_extensions.shortest_path_follower import ShortestPathFollowerCompat
from habitat_extensions.utils import asnumpy
from vlnce_baselines.agents.r2r_envdrop.utils import Tokenizer

@registry.register_sensor(name="InstructionSensor")
class InstructionSensor(Sensor):
    r"""The agent instruction sensor. 
    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions
                to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._dimensionality = getattr(config, "DIMENSIONALITY", 200)
        super().__init__(config=config)
        
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "instruction"
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TEXT
    
    def _get_observation(
        self, observations: Dict[str, Observations], episode: VLNEpisode,
        **kwargs
    ):
        return {
            "text": episode.instruction.instruction_text,
            "tokens": episode.instruction.instruction_tokens,
            "trajectory_id": episode.trajectory_id,
        }

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1, self._dimensionality),
            dtype=np.float32,
        )

@registry.register_sensor(name="GlobalGPSSensor")
class GlobalGPSSensor(Sensor):
    r"""The agents current location in the global coordinate frame

    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions
                to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "globalgps"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._dimensionality,),
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode, **kwargs: Any):
        agent_position = self._sim.get_agent_state().position
        return agent_position.astype(np.float32)

@registry.register_sensor(name="GlobalCompassSensor")
class GlobalCompassSensor(Sensor):
    r"""The agents current rotation in the global coordinate frame

    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions
                to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._dimensionality = getattr(config, "DIMENSIONALITY", 4)
        # assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "globalcompass"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._dimensionality,),
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode, **kwargs: Any):
        agent_rotation = self._sim.get_agent_state().rotation
        agent_rotation = np.array(
            quaternion_to_list(agent_rotation), dtype=np.float32)
        # return agent_rotation.astype(np.float32)
        return agent_rotation


@registry.register_sensor
class OracleActionSensor(Sensor):
    r"""Sensor for observing the optimal action to take. The assumption this
    sensor currently makes is that the shortest path to the goal is the
    optimal path.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)
        self.follower = ShortestPathFollower(
            self._sim,
            # all goals can be navigated to within 0.5m.
            goal_radius=getattr(config, "GOAL_RADIUS", 0.5),
            return_one_hot=False,
        )
        self.follower.mode = "geodesic_path"

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "oracle_action_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0, 
            high=100, 
            shape=(1,), 
            dtype=np.float
        )

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        best_action = self.follower.get_next_action(episode.goals[0].position)
        return np.array(
            [best_action if best_action is not None else HabitatSimActions.STOP]
        )

@registry.register_sensor
class OracleObservationSensor(Sensor):
    r"""Sensor for observing the optimal view.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)
        self.follower = ShortestPathFollower(
            self._sim,
            # all goals can be navigated to within 0.5m.
            goal_radius=getattr(config, "GOAL_RADIUS", 0.5),
            return_one_hot=False,
        )
        self.follower.mode = "geodesic_path"

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "oracle_observation_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0, 
            high=100, 
            shape=(1,), 
            dtype=np.float
        )

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        best_action = self.follower.get_next_action(episode.goals[0].position)
        return self._sim.step(best_action)


@registry.register_sensor
class OracleProgressSensor(Sensor):
    r"""Sensor for observing how much progress has been made towards the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "progress"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        # TODO: what is the correct sensor type?
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float
        )

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        distance_from_start = episode.info["geodesic_distance"]

        return (distance_from_start - distance_to_target) / distance_from_start

@registry.register_sensor
class ProximitySensor(Sensor):
    r"""Sensor for observing the distance to the closest obstacle
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    def __init__(self, sim, config):
        self._sim = sim
        self._max_detection_radius = getattr(
            config, "MAX_DETECTION_RADIUS", 2.0
        )
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "proximity"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0,
            high=self._max_detection_radius,
            shape=(1,),
            dtype=np.float,
        )

    def get_observation(self, observations, episode):
        current_position = self._sim.get_agent_state().position

        return self._sim.distance_to_closest_obstacle(
            current_position, self._max_detection_radius
        )

@registry.register_sensor
class RelativePointGoalSensor(Sensor):
    r""" Relative pointgoal sensor """
    cls_uuid: str = "rel_pointgoal"
    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,), 
            dtype=np.float32
        )

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        start = np.array([0, 0])
        d = episode.goals[0].rel_distance
        h = episode.goals[0].rel_heading
        e = episode.goals[0].rel_elevation
        # goal  = np.array([d, d * np.cos(h), d * np.sin(h)])
        return np.array([d, h], dtype=np.float)

@registry.register_sensor
class VLNOracleActionSensor(Sensor):
    r"""Sensor for observing the optimal action to take. The assumption this
    sensor currently makes is that the shortest path to the goal is the
    optimal path.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        super().__init__(config=config)

        # all goals can be navigated to within 0.5m.
        goal_radius = getattr(config, "GOAL_RADIUS", 0.5)
        if config.USE_ORIGINAL_FOLLOWER:
            self.follower = ShortestPathFollowerCompat(
                sim, goal_radius, return_one_hot=False
            )
            self.follower.mode = "geodesic_path"
        else:
            self.follower = ShortestPathFollower(
                sim, goal_radius, return_one_hot=False
            )

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "vln_oracle_action_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0, 
            high=100, 
            shape=(1,), 
            dtype=np.float
        )

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        best_action = self.follower.get_next_action(episode.goals[0].position)
        return np.array(
            [best_action if best_action is not None else HabitatSimActions.STOP]
        )


@registry.register_sensor
class VLNOracleProgressSensor(Sensor):
    r"""Sensor for observing how much progress has been made towards the goal.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "progress"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        # TODO: what is the correct sensor type?
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float
        )

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        distance_from_start = episode.info["geodesic_distance"]

        return (distance_from_start - distance_to_target) / distance_from_start


class MapPlaceHolder(Sensor):
    r""" Placeholder for map observations.
    From the Learning to set Waypoints paper: https://arxiv.org/pdf/2008.09622.pdf
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, 
        **kwargs: Any
    ):
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(self.config.MAP_SIZE, self.config.MAP_SIZE, self.config.NUM_CHANNEL),
            dtype=np.uint8,
        )

    def get_observation(
            self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        return np.zeros((
            self.config.MAP_SIZE, self.config.MAP_SIZE, self.config.NUM_CHANNEL
        ))

@registry.register_sensor(name="GeometricMap")
class GeometricMap(MapPlaceHolder):
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "gm"

@registry.register_sensor(name="ActionMap")
class ActionMap(MapPlaceHolder):
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "action_map"

@registry.register_sensor(name="Collision")
class Collision(Sensor):
    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any,
        **kwargs: Any
    ):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "collision"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=bool
        )

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        return [self._sim.previous_step_collided]


@registry.register_sensor(name="EgoMap")
class EgoMap(Sensor):
    r"""Estimates the top-down occupancy based on current depth-map.
    From the Learning to set Waypoints paper: https://arxiv.org/pdf/2008.09622.pdf
    Args:
        sim: reference to the simulator for calculating task observations.
        config: contains the MAP_RESOLUTION, MAP_SIZE, HEIGHT_THRESH fields to
                decide grid-size, extents of the projection, and the thresholds
                for determining obstacles and explored space.
    """

    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim

        super().__init__(config=config)

        # Map statistics
        self.map_size = self.config.MAP_SIZE
        self.map_res = self.config.MAP_RESOLUTION

        # Agent height for pointcloud transformation
        self.sensor_height = self.config.POSITION[1]

        # Compute intrinsic matrix
        hfov = float(self._sim.config.DEPTH_SENSOR.HFOV) * np.pi / 180
        self.intrinsic_matrix = np.array([[1 / np.tan(hfov / 2.), 0., 0., 0.],
                                          [0., 1 / np.tan(hfov / 2.), 0., 0.],
                                          [0., 0.,  1, 0],
                                          [0., 0., 0, 1]])
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)

        # Height thresholds for obstacles
        self.height_thresh = self.config.HEIGHT_THRESH

        # Depth processing
        self.min_depth = float(self._sim.config.DEPTH_SENSOR.MIN_DEPTH)
        self.max_depth = float(self._sim.config.DEPTH_SENSOR.MAX_DEPTH)

        # Pre-compute a grid of locations for depth projection
        W = self._sim.config.DEPTH_SENSOR.WIDTH
        H = self._sim.config.DEPTH_SENSOR.HEIGHT
        self.proj_xs, self.proj_ys = np.meshgrid(
                                          np.linspace(-1, 1, W),
                                          np.linspace(1, -1, H)
                                     )

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "ego_map"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self.config.MAP_SIZE, self.config.MAP_SIZE, 2)
        return spaces.Box(
            low=0,
            high=1,
            shape=sensor_shape,
            dtype=np.uint8,
        )

    def convert_to_pointcloud(self, depth):
        """
        Inputs:
            depth = (H, W, 1) numpy array
        Returns:
            xyz_camera = (N, 3) numpy array for (X, Y, Z) in egocentric world 
            coordinates
        """

        depth_float = depth.astype(np.float32)[..., 0]

        # =========== Convert to camera coordinates ============
        W = depth.shape[1]
        xs = self.proj_xs.reshape(-1)
        ys = self.proj_ys.reshape(-1)
        depth_float = depth_float.reshape(-1)

        # Filter out invalid depths
        max_forward_range = self.map_size * self.map_res
        valid_depths = (depth_float != 0.0) & (depth_float <= max_forward_range)
        xs = xs[valid_depths]
        ys = ys[valid_depths]
        depth_float = depth_float[valid_depths]

        # Unproject
        # negate depth as the camera looks along -Z
        xys = np.vstack((xs * depth_float,
                         ys * depth_float,
                         -depth_float, np.ones(depth_float.shape)))
        inv_K = self.inverse_intrinsic_matrix
        xyz_camera = np.matmul(inv_K, xys).T # XYZ in the camera coordinate system
        xyz_camera = xyz_camera[:, :3] / xyz_camera[:, 3][:, np.newaxis]

        return xyz_camera

    def safe_assign(self, im_map, x_idx, y_idx, value):
        try:
            im_map[x_idx, y_idx] = value
        except IndexError:
            valid_idx1 = np.logical_and(x_idx >= 0, x_idx < im_map.shape[0])
            valid_idx2 = np.logical_and(y_idx >= 0, y_idx < im_map.shape[1])
            valid_idx = np.logical_and(valid_idx1, valid_idx2)
            im_map[x_idx[valid_idx], y_idx[valid_idx]] = value

    def _get_depth_projection(self, sim_depth):
        """
        Project pixels visible in depth-map to ground-plane
        """
        if self._sim.config.DEPTH_SENSOR.NORMALIZE_DEPTH:
            depth = sim_depth * (self.max_depth - self.min_depth) + self.min_depth
        else:
            depth = sim_depth

        XYZ_ego = self.convert_to_pointcloud(depth)

        # Adding agent's height to the point cloud
        XYZ_ego[:, 1] += self.sensor_height

        # Convert to grid coordinate system
        V = self.map_size
        Vby2 = V // 2
        points = XYZ_ego

        grid_x = (points[:, 0] / self.map_res) + Vby2
        grid_y = (points[:, 2] / self.map_res) + V

        # Filter out invalid points
        valid_idx = (grid_x >= 0) & (grid_x <= V-1) & (grid_y >= 0) & (grid_y <= V-1)
        points = points[valid_idx, :]
        grid_x = grid_x[valid_idx].astype(int)
        grid_y = grid_y[valid_idx].astype(int)

        # Create empty maps for the two channels
        obstacle_mat = np.zeros((self.map_size, self.map_size), np.uint8)
        explore_mat = np.zeros((self.map_size, self.map_size), np.uint8)

        # Compute obstacle locations
        high_filter_idx = points[:, 1] < self.height_thresh[1]
        low_filter_idx = points[:, 1] > self.height_thresh[0]
        obstacle_idx = np.logical_and(low_filter_idx, high_filter_idx)

        self.safe_assign(obstacle_mat, grid_y[obstacle_idx], grid_x[obstacle_idx], 1)

        # Compute explored locations
        explored_idx = high_filter_idx
        self.safe_assign(explore_mat, grid_y[explored_idx], grid_x[explored_idx], 1)

        # Smoothen the maps
        kernel = np.ones((3, 3), np.uint8)

        obstacle_mat = cv2.morphologyEx(obstacle_mat, cv2.MORPH_CLOSE, kernel)
        explore_mat = cv2.morphologyEx(explore_mat, cv2.MORPH_CLOSE, kernel)

        # Ensure all expanded regions in obstacle_mat are accounted for in explored_mat
        explore_mat = np.logical_or(explore_mat, obstacle_mat)

        return np.stack([obstacle_mat, explore_mat], axis=2)

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        
        sim_depth = asnumpy(observations['depth'])
        ego_map_gt = self._get_depth_projection(sim_depth)
            
        # # convert to numpy array
        # ego_map_gt = self._sim.get_egomap_observation()
        
        # if ego_map_gt is None:
        #     sim_depth = asnumpy(observations['depth'])
        #     ego_map_gt = self._get_depth_projection(sim_depth)
        #     self._sim.cache_egomap_observation(ego_map_gt)

        return ego_map_gt


def asnumpy(v):
    if torch.is_tensor(v):
        return v.cpu().numpy()
    elif isinstance(v, np.ndarray):
        return v
    else:
        raise ValueError('Invalid input')
    
@registry.register_sensor(name="TemplateSensor")
class TemplateSensor(Sensor):
    def __init__(self, sim: Simulator,config:Config, **kwargs):
        self._sim = sim
        self.uuid = "template_sensor"
        self.observation_space = spaces.Discrete(0)
        vocab_path = config.VOCAB_PATH
        with gzip.open(vocab_path, 'r') as fin:
            data = json.loads(fin.read().decode('utf-8'))
        vocab = data['instruction_vocab']['word_list']

        self.tok = Tokenizer(vocab=vocab, encoding_length=config.MAX_LENGTH)
        self.cat2vocab = json.load(open(config.CAT_MAPPER))
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation(
        self,
        observations,
        episode: Episode,
        **kwargs
    ): 
        # if "semantic" not in observations.keys():
        #     return {
        #     "text": "start",
        #     "tokens": tok.encode_sentence("start"),
        # }
        semantic_obs = observations["semantic"]
        scene = self._sim.semantic_annotations()
        counts = np.bincount(semantic_obs[40:220,60:200].flatten())
        leftcounts = np.bincount(semantic_obs[:,0:80].flatten())
        rightcounts = np.bincount(semantic_obs[:,160:].flatten())
        total_count = np.sum(counts)
        total_lcount = np.sum(leftcounts)
        total_rcount = np.sum(rightcounts)
        exclude = ['wall','ceiling','floor','misc','void']
        max_pix = 0
        for object_i, count in enumerate(counts):
            sem_obj = scene.objects[object_i]
            cat = sem_obj.category.name()
            pixel_ratio = count / total_count
            if pixel_ratio > max_pix and cat not in exclude:
                max_pix = pixel_ratio
                max_obj = cat
        if max_pix is 0:
            max_pix = 0
            for object_i, count in enumerate(counts):
                sem_obj = scene.objects[object_i]
                cat = sem_obj.category.name()
                pixel_ratio = count / total_count
                if pixel_ratio > max_pix:
                    max_pix = pixel_ratio
                    max_obj = cat
        max_pix = 0
        for object_i, count in enumerate(leftcounts):
            sem_obj = scene.objects[object_i]
            cat = sem_obj.category.name()
            pixel_ratio = count / total_lcount
            if pixel_ratio > max_pix and cat not in exclude:
                max_pix = pixel_ratio
                left_obj = cat
        if max_pix is 0:
            max_pix = 0
            for object_i, count in enumerate(leftcounts):
                sem_obj = scene.objects[object_i]
                cat = sem_obj.category.name()
                pixel_ratio = count / total_lcount
                if pixel_ratio > max_pix :
                    max_pix = pixel_ratio
                    left_obj = cat
        max_pix = 0
        for object_i, count in enumerate(rightcounts):
            sem_obj = scene.objects[object_i]
            cat = sem_obj.category.name()
            pixel_ratio = count / total_rcount
            if pixel_ratio > max_pix and cat not in exclude:
                max_pix = pixel_ratio
                right_obj = cat
        if max_pix is 0:
            max_pix = 0
            for object_i, count in enumerate(rightcounts):
                sem_obj = scene.objects[object_i]
                cat = sem_obj.category.name()
                pixel_ratio = count / total_rcount
                if pixel_ratio > max_pix:
                    max_pix = pixel_ratio
                    right_obj = cat
        max_obj = self.cat2vocab.get(max_obj,"misc")
        left_obj = self.cat2vocab.get(left_obj,"misc")
        right_obj = self.cat2vocab.get(right_obj,"misc")
        if max_obj == left_obj == right_obj:
            string = f"A {max_obj} is in front."  
        elif max_obj == left_obj:
            string = f"A {max_obj} is in front and to my left. A {right_obj} is to my right." 
        elif max_obj == right_obj:
            string = f"A {max_obj} is in front and to my right. A {left_obj} is to my left."
        elif left_obj == right_obj:
            string = f"A {max_obj} is in front. A {left_obj} is to my left and to my right."  
        else:
            string  = f"A {max_obj} is in front a {left_obj} to my left and a {right_obj} to my right."
        tokenized = self.tok.encode_sentence(string)
        #print(string)
        return {
            "text": string,
            "tokens": tokenized,
        }
        
    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)
        # ego_map_gt = self._sim.get_egomap_observation()
        # self._sim.cache_egomap_observation(ego_map_gt)
        # ego_map_gt = self._sim.get_egomap_observation()
        # if ego_map_gt is None:
        #     sim_depth = asnumpy(observations['depth'])
        #     ego_map_gt = self._get_depth_projection(sim_depth)
        #     self._sim.cache_egomap_observation(ego_map_gt)
        sim_depth = asnumpy(observations['depth'])
        ego_map_gt = self._get_depth_projection(sim_depth)
        return ego_map_gt
    
    
@registry.register_sensor(name="LxmertInstructionSensor")
class LxmertInstructionSensor(Sensor):
    def __init__(self, config: Config, **kwargs):
        self.uuid = "lxmertinstruction"
        self.observation_space = spaces.Discrete(0)
        self.max_length = getattr(config, "MAX_LENGTH", 160)
        self.tokenizer = LxmertTokenizer.from_pretrained(
            getattr(config, "TOKENIZER", 'unc-nlp/lxmert-base-uncased')
        )
        
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation(
        self, observations: Dict[str, Observations], episode: VLNEpisode,
        **kwargs
    ):
        tokens = self.tokenizer(
            episode.instruction.instruction_text, 
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        return {
            "text": episode.instruction.instruction_text,
            "input_ids": tokens['input_ids'],
            "token_type_ids": tokens['token_type_ids'],
            "attention_mask": tokens['attention_mask'],
            "trajectory_id": episode.trajectory_id,

        }

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)

@registry.register_sensor(name="ReasoningSensor")
class ReasoningSensor(Sensor):
    def __init__(self, sim: Simulator, config:Config, **kwargs):
        self._sim = sim
        self.uuid = "reasoning_sensor"
        self.observation_space = spaces.Discrete(0)
        self._config = config
        self.kg = json.load(open(getattr(config,'KG_FILE','data/models/knowledge_graph.json')))
        self.obj_list = json.load(open(getattr(config,'OBJ_LIST','data/models/objcat.json')))

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.uuid

    def _get_observation(
        self, observations: Dict[str, Observations], episode: VLNEpisode,
        **kwargs
    ):
        state = self._sim.get_agent_state()
        scene = self._sim.semantic_annotations()
        regions = scene.regions

        room = run_exp(state,regions)
        instruction_objects = episode.objects
        semantic_obs = observations["semantic"]
        counts = np.bincount(semantic_obs.flatten())
        scene_objects = []
        kg_objects = []
        max_occ = 0
        return_obj = 'misc'

        for id, count in enumerate(counts):
            if count != 0:
                obj = scene.objects[id].category.name()
                scene_objects.append(obj)
                if room in self.kg.get(obj, []):
                    kg_objects.append((obj, room, self.kg.get(obj).get(room,0)))
                if obj in self.kg.get(room,[]) :
                    kg_objects.append((obj, room, self.kg.get(room).get(obj,0)))

        if not kg_objects:
            kg_objects.append(('misc','no label',1))

        for obj in scene_objects:
            for instruction_object in instruction_objects:
                if obj == instruction_object[0]:
                    return torch.tensor(self.obj_list.index(obj))
                if obj == instruction_object[1] and max_occ < instruction_object[2]:
                    max_occ = instruction_object[2]
                    return_obj = obj
        obj, _, value = max(kg_objects, key = itemgetter(2))
        if max_occ < value:
            return_obj = obj
        if return_obj == 'misc':
            return_obj = random.choice(scene_objects)
        try :
            idx = self.obj_list.index(return_obj)
        except ValueError:
            idx = 39

        return torch.tensor(idx) 

    def get_observation(self, **kwargs):
        return self._get_observation(**kwargs)