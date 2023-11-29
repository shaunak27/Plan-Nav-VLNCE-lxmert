# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional
from abc import ABC
from collections import defaultdict, namedtuple
import logging
import time
import pickle
import os

import scipy
from scipy.io import wavfile
from scipy.signal import fftconvolve
import numpy as np
from scipy.spatial import distance
import networkx as nx
from gym import spaces

from habitat.core.spaces import Space
from habitat.core.dataset import Episode
from habitat.core.registry import registry
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_angle_axis
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.core.simulator import (
    AgentState,
    Config,
    Observations,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
)


def load_metadata(parent_folder):
    points_file = os.path.join(parent_folder, 'points.txt')
    if "replica" in parent_folder:
        graph_file = os.path.join(parent_folder, 'graph.pkl')
        points_data = np.loadtxt(points_file, delimiter="\t")
        points = list(zip(
            points_data[:, 1],
            points_data[:, 3] - 1.5528907,
            -points_data[:, 2])
        )
    else:
        graph_file = os.path.join(parent_folder, 'graph.pkl')
        points_data = np.loadtxt(points_file, delimiter="\t")
        points = list(zip(
            points_data[:, 1],
            points_data[:, 3] - 1.5,
            -points_data[:, 2])
        )
    if not os.path.exists(graph_file):
        raise FileExistsError(graph_file + ' does not exist!')
    else:
        with open(graph_file, 'rb') as fo:
            graph = pickle.load(fo)

    return points, graph


def overwrite_config(config_from: Config, config_to: Any) -> None:
    r"""Takes Habitat-API config and Habitat-Sim config structures. Overwrites
    Habitat-Sim config with Habitat-API values, where a field name is present
    in lowercase. Mostly used to avoid :ref:`sim_cfg.field = hapi_cfg.FIELD`
    code.
    Args:
        config_from: Habitat-API config node.
        config_to: Habitat-Sim config structure.
    """

    def if_config_to_lower(config):
        if isinstance(config, Config):
            return {key.lower(): val for key, val in config.items()}
        else:
            return config

    for attr, value in config_from.items():
        if hasattr(config_to, attr.lower()):
            setattr(config_to, attr.lower(), if_config_to_lower(value))


class DummySimulator:
    """
    Dummy simulator for avoiding loading the scene meshes when using cached observations.
    """
    def __init__(self):
        self.position = None
        self.rotation = None
        self._sim_obs = None

    def seed(self, seed):
        pass

    def set_agent_state(self, position, rotation):
        self.position = np.array(position, dtype=np.float32)
        self.rotation = rotation

    def get_agent_state(self):
        class State:
            def __init__(self, position, rotation):
                self.position = position
                self.rotation = rotation

        return State(self.position, self.rotation)

    def set_sensor_observations(self, sim_obs):
        self._sim_obs = sim_obs

    def get_sensor_observations(self):
        return self._sim_obs

    def close(self):
        pass


@registry.register_simulator()
class DwpSim(Simulator, ABC):
    r"""Changes made to simulator wrapper over habitat-sim.
    This simulator first loads the graph of current environment and moves the 
    agent among nodes. Any sounds can be specified in the episode and loaded in 
    this simulator.
    Args:
        config: configuration for initializing the simulator.
    """

    def action_space_shortest_path(
        self, source: AgentState, targets: List[AgentState], agent_id: int = 0
    ) -> List[ShortestPathPoint]:
        pass

    def __init__(self, config: Config) -> None:
        self.config = config
        agent_config = self._get_agent_config()
        sim_sensors = []
        for sensor_name in agent_config.SENSORS:
            sensor_cfg = getattr(self.config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            sim_sensors.append(sensor_type(sensor_cfg))

        self._sensor_suite = SensorSuite(sim_sensors)
        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._current_scene = self.sim_config.sim_cfg.scene.id
        self._action_space = spaces.Discrete(
            len(self.sim_config.agents[0].action_space)
        )
        self._prev_sim_obs = None

        self._source_position_index = None
        self._receiver_position_index = None
        self._rotation_angle = None
        self._current_sound = None
        self._offset = None
        self._duration = None
        self._frame_cache = dict()
        self._spectrogram_cache = dict()
        self._egomap_cache = defaultdict(dict)
        self._scene_observations = None
        self._episode_step_count = None
        self._is_episode_active = None
        self._position_to_index_mapping = dict()
        self._previous_step_collided = False
        self._house_readers = dict()
        self._use_oracle_planner = True
        self._oracle_actions = list()
        self._all_position_encodings = None

        self.points, self.graph = load_metadata(self.metadata_dir)
        for node in self.graph.nodes():
            self._position_to_index_mapping[self.position_encoding(
                self.graph.nodes()[node]['point']
            )] = node
        self._all_position_encodings = self._parse_position_encodings(
            list(self._position_to_index_mapping.keys())
        )

        self._sim = habitat_sim.Simulator(config=self.sim_config)

    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        overwrite_config(
            config_from=self.config.HABITAT_SIM_V0, config_to=sim_config
        )
        sim_config.scene.id = self.config.SCENE
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=self.get_agent_config(), config_to=agent_config
        )

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            sim_sensor_cfg = habitat_sim.SensorSpec()
            overwrite_config(
                config_from=sensor.config, config_to=sim_sensor_cfg
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(
                sensor.observation_space.shape[:2]
            )
            sim_sensor_cfg.parameters["hfov"] = str(sensor.config.HFOV)

            # accessing child attributes through parent interface
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type  # type: ignore
            sim_sensor_cfg.gpu2gpu_transfer = (
                self.config.HABITAT_SIM_V0.GPU_GPU
            )
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = registry.get_action_space_configuration(
            self.config.ACTION_SPACE_CONFIG
        )(self.config).get()

        return habitat_sim.Configuration(sim_config, [agent_config])

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    def get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.config.DEFAULT_AGENT_ID
        agent_name = self.config.AGENTS[agent_id]
        agent_config = getattr(self.config, agent_name)
        return agent_config

    def _parse_position_encodings(self, position_encodings):
        position_encodings_arr = []
        for p in position_encodings:
            x, y, z = self.decode_position_encoding(p)
            position_encodings_arr.append([x, y, z])
        return np.array(position_encodings_arr)

    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, _ in enumerate(self.config.AGENTS):
            agent_cfg = self._get_agent_config(agent_id)
            if agent_cfg.IS_SET_START_STATE:
                self.set_agent_state(
                    agent_cfg.START_POSITION,
                    agent_cfg.START_ROTATION,
                    agent_id,
                )
                is_updated = True

        return is_updated

    def _get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.config.DEFAULT_AGENT_ID
        agent_name = self.config.AGENTS[agent_id]
        agent_config = getattr(self.config, agent_name)
        return agent_config

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        return self._sim.get_agent(agent_id).get_state()

    def set_agent_state(
        self, position: List[float], rotation: List[float], agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        # if self.config.USE_RENDERED_OBSERVATIONS:
        #     self._sim.set_agent_state(position, rotation)
        # else:
        agent = self._sim.get_agent(agent_id)
        new_state = self.get_agent_state(agent_id)
        new_state.position = position
        new_state.rotation = rotation

        # NB: The agent state also contains the sensor states in _absolute_
        # coordinates. In order to set the agent's body to a specific
        # location and have the sensors follow, we must not provide any
        # state for the sensors. This will cause them to follow the agent's
        # body
        new_state.sensor_states = {}
        agent.set_state(new_state, reset_sensors)
        return True

    @property
    def metadata_dir(self):
        return os.path.join(
            self.config.METADATA_DIR,
            self.config.SCENE_DATASET,
            self.current_scene_name
        )

    @property
    def current_scene_name(self):
        # config.SCENE (_current_scene) looks like
        # 'data/scene_datasets/replica/office_1/habitat/mesh_semantic.ply'
        return self._current_scene.split('/')[3]

    @property
    def current_scene_observation_file(self):
        return os.path.join(
            self.config.SCENE_OBSERVATION_DIR, self.config.SCENE_DATASET,
            self.current_scene_name + '.pkl'
        )

    @property
    def pathfinder(self):
        return self._sim.pathfinder

    def get_agent(self, agent_id):
        return self._sim.get_agent(agent_id)

    def reconfigure(self, config: Config) -> None:
        self.config = config
        if hasattr(self.config.AGENT_0, 'OFFSET'):
            self._offset = int(self.config.AGENT_0.OFFSET)
        else:
            self._offset = 0

        self._duration = 500

        is_same_scene = config.SCENE == self._current_scene
        if not is_same_scene:
            self._current_scene = config.SCENE
            logging.debug('Current scene: {} and sound: {}'.format(
                self.current_scene_name, self._current_sound))

            self._sim.close()
            del self._sim
            self.sim_config = self.create_sim_config(self._sensor_suite)
            self._sim = habitat_sim.Simulator(self.sim_config)
            self._update_agents_state()
            self._frame_cache = dict()
            logging.debug('Loaded scene {}'.format(self.current_scene_name))

            self.points, self.graph = load_metadata(self.metadata_dir)
            for node in self.graph.nodes():
                self._position_to_index_mapping[self.position_encoding(
                    self.graph.nodes()[node]['point'])] = node

        self._episode_step_count = 0

        # set agent positions
        self._receiver_position_index = self._position_to_index(
            self.config.AGENT_0.START_POSITION
        )
        self._source_position_index = self._position_to_index(
            self.config.AGENT_0.GOAL_POSITION
        )
        # the agent rotates about +Y starting from -Z counterclockwise,
        # so rotation angle 90 means the agent rotate about +Y 90 degrees
        self._rotation_angle = int(np.around(np.rad2deg(quat_to_angle_axis(
            quat_from_coeffs(self.config.AGENT_0.START_ROTATION))[0]))) % 360


        self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                            self.config.AGENT_0.START_ROTATION)

        # if self._use_oracle_planner:
        #     self._oracle_actions = self.compute_oracle_actions()

        logging.debug("Initial source, agent at: {}, {}, orientation: {}".
                      format(self._source_position_index, self._receiver_position_index, self.get_orientation()))

    def compute_semantic_index_mapping(self):
        raise NotImplementedError("not implemented")

    @staticmethod
    def position_encoding(position):
        if len(position) == 1:
            position = position[0]
        return '{:.2f}_{:.2f}_{:.2f}'.format(*position)

    @staticmethod
    def decode_position_encoding(position_encoding):
        return [eval(i) for i in position_encoding.split("_")]

    def _position_to_index(self, position):
        if self.position_encoding(position) in self._position_to_index_mapping:
            return self._position_to_index_mapping[self.position_encoding(position)]
        else:
            # compute euclidean distance between the vlnce starting point and the grid point
            closest_position_encoding = self._get_closest_position_encoding(
                self.position_encoding(position))
            return self._position_to_index_mapping[closest_position_encoding]
            # raise ValueError("Position misalignment.")

    @staticmethod
    def closest_node(node, nodes):
        closest_index = distance.cdist([node], nodes).argmin()
        return nodes[closest_index]

    def _get_closest_position_encoding(self, random_position_encoding):
        x, y, z = self.decode_position_encoding(random_position_encoding)
        random_position = np.array([x, y, z])
        closest_position = self.closest_node(
            random_position, self._all_position_encodings)
        # change back to string format
        closest_position_encoding = self.position_encoding(closest_position.tolist())
        print(
            f"original vlnce point: {random_position_encoding}\n "
            f"mapped closest point: {closest_position_encoding}")
        return closest_position_encoding

    def _get_sim_observation(self):
        joint_index = (self._receiver_position_index, self._rotation_angle)
        if joint_index in self._frame_cache:
            return self._frame_cache[joint_index]
        else:
            # assert not self.config.USE_RENDERED_OBSERVATIONS
            sim_obs = self._sim.get_sensor_observations()
            for sensor in sim_obs:
                sim_obs[sensor] = sim_obs[sensor]
            self._frame_cache[joint_index] = sim_obs
            return sim_obs

    def reset(self):
        logging.debug('Reset simulation')
        # if self.config.USE_RENDERED_OBSERVATIONS:
        #     sim_obs = self._get_sim_observation()
        #     self._sim.set_sensor_observations(sim_obs)
        # else:
        if 1:
            sim_obs = self._sim.reset()
            if self._update_agents_state():
                sim_obs = self._get_sim_observation()

        self._is_episode_active = True
        self._prev_sim_obs = sim_obs
        self._previous_step_collided = False
        # Encapsule data under Observations class
        observations = self._sensor_suite.get_observations(sim_obs)

        return observations

    def step(self, action, only_allowed=True):
        """
        All angle calculations in this function is w.r.t habitat coordinate 
        frame, on X-Z plane where +Y is upward, -Z is forward and +X is rightward.
        Angle 0 corresponds to +X, angle 90 corresponds to +y and 290 corresponds 
        to 270.

        :param action: action to be taken
        :param only_allowed: if true, then can't step anywhere except allowed 
        locations
        :return:
        Dict of observations
        """
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )

        self._previous_step_collided = False
        # STOP: 0, FORWARD: 1, LEFT: 2, RIGHT: 2
        if action == HabitatSimActions.STOP:
            self._is_episode_active = False
        else:
            prev_position_index = self._receiver_position_index
            prev_rotation_angle = self._rotation_angle
            if action == HabitatSimActions.MOVE_FORWARD:
                # the agent initially faces -Z by default
                self._previous_step_collided = True
                for neighbor in self.graph[self._receiver_position_index]:
                    p1 = self.graph.nodes[self._receiver_position_index]['point']
                    p2 = self.graph.nodes[neighbor]['point']
                    direction = int(np.around(
                        np.rad2deg(np.arctan2(p2[2] - p1[2], p2[0] - p1[0])))) % 360
                    if direction == self.get_orientation():
                        self._receiver_position_index = neighbor
                        self._previous_step_collided = False
                        break
            elif action == HabitatSimActions.TURN_LEFT:
                # agent rotates counterclockwise, so turning left means increasing rotation angle by 90
                self._rotation_angle = (self._rotation_angle + 90) % 360
            elif action == HabitatSimActions.TURN_RIGHT:
                self._rotation_angle = (self._rotation_angle - 90) % 360

            if self.config.CONTINUOUS_VIEW_CHANGE:
                intermediate_observations = list()
                fps = self.config.VIEW_CHANGE_FPS
                if action == HabitatSimActions.MOVE_FORWARD:
                    prev_position = np.array(
                        self.graph.nodes[prev_position_index]['point'])
                    current_position = np.array(
                        self.graph.nodes[self._receiver_position_index]['point'])
                    for i in range(1, fps):
                        intermediate_position = prev_position + i / fps * (current_position - prev_position)
                        self.set_agent_state(intermediate_position.tolist(), quat_from_angle_axis(np.deg2rad(
                                            self._rotation_angle), np.array([0, 1, 0])))
                        sim_obs = self._sim.get_sensor_observations()
                        observations = self._sensor_suite.get_observations(sim_obs)
                        intermediate_observations.append(observations)
                else:
                    for i in range(1, fps):
                        if action == HabitatSimActions.TURN_LEFT:
                            intermediate_rotation = prev_rotation_angle + i / fps * 90
                        elif action == HabitatSimActions.TURN_RIGHT:
                            intermediate_rotation = prev_rotation_angle - i / fps * 90
                        self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                             quat_from_angle_axis(np.deg2rad(intermediate_rotation),
                                                                  np.array([0, 1, 0])))
                        sim_obs = self._sim.get_sensor_observations()
                        observations = self._sensor_suite.get_observations(sim_obs)
                        intermediate_observations.append(observations)

            self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                                 quat_from_angle_axis(np.deg2rad(self._rotation_angle), np.array([0, 1, 0])))
        self._episode_step_count += 1

        # log debugging info
        logging.debug('After taking action {}, s,r: {}, {}, orientation: {}, location: {}'.format(
            action, self._source_position_index, self._receiver_position_index,
            self.get_orientation(), self.graph.nodes[self._receiver_position_index]['point']))

        sim_obs = self._get_sim_observation()
        # if self.config.USE_RENDERED_OBSERVATIONS:
        #     self._sim.set_sensor_observations(sim_obs)

        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)
        if self.config.CONTINUOUS_VIEW_CHANGE:
            observations['intermediate'] = intermediate_observations

        return observations

    def get_orientation(self):
        _base_orientation = 270
        return (_base_orientation - self._rotation_angle) % 360

    # def sample_navigable_point(self):
    #     return self._sim.pathfinder.get_random_navigable_point().tolist()

    def is_navigable(self, point: List[float]):
        return self._sim.pathfinder.is_navigable(point)

    @property
    def azimuth_angle(self):
        # this is the angle used to index the binaural audio files
        # in mesh coordinate systems, +Y forward, +X rightward, +Z upward
        # azimuth is calculated clockwise so +Y is 0 and +X is 90
        return -(self._rotation_angle + 0) % 360

    @property
    def reaching_goal(self):
        return self._source_position_index == self._receiver_position_index

    def _compute_euclidean_distance_between_sr_locations(self):
        p1 = self.graph.nodes[self._receiver_position_index]['point']
        p2 = self.graph.nodes[self._source_position_index]['point']
        d = np.sqrt((p1[0] - p2[0])**2 + (p1[2] - p2[2])**2)
        return d

    def get_egomap_observation(self):
        joint_index = (self._receiver_position_index, self._rotation_angle)
        if joint_index in self._egomap_cache[self._current_scene]:
            return self._egomap_cache[self._current_scene][joint_index]
        else:
            return None

    def cache_egomap_observation(self, egomap):
        self._egomap_cache[self._current_scene][(self._receiver_position_index, self._rotation_angle)] = egomap

    def geodesic_distance(self, position_a, position_bs, episode=None):
        # print(f"position a: {position_a} position_bs: {position_bs}")
        index_a = self._position_to_index(position_a)
        index_b = self._position_to_index(position_bs)
        assert index_a is not None and index_b is not None
        path_length = nx.shortest_path_length(
            self.graph, index_a, index_b) * self.config.GRID_SIZE
        return path_length

        # print("geodesic distance:")
        # print(f"position_a: {position_a}")
        # print(f"position_bs: {position_bs}")
        # distances = []
        # for position_b in position_bs:
        #     index_a = self._position_to_index(position_a)
        #     index_b = self._position_to_index(position_b)
        #     assert index_a is not None and index_b is not None
        #     path_length = nx.shortest_path_length(self.graph, index_a, index_b) * self.config.GRID_SIZE
        #     distances.append(path_length)
        #
        # return min(distances)

    @property
    def forward_vector(self):
        return -np.array([0.0, 0.0, 1.0])

    def get_straight_shortest_path_points(self, position_a, position_b):
        index_a = self._position_to_index(position_a)
        index_b = self._position_to_index(position_b)
        assert index_a is not None and index_b is not None

        shortest_path = nx.shortest_path(
            self.graph, source=index_a, target=index_b)
        points = list()
        for node in shortest_path:
            points.append(self.graph.nodes()[node]['point'])
        return points

    def compute_oracle_actions(self):
        start_node = self._receiver_position_index
        end_node = self._source_position_index
        shortest_path = nx.shortest_path(
            self.graph, source=start_node, target=end_node)
        assert shortest_path[0] == start_node and shortest_path[-1] == end_node
        logging.debug(shortest_path)

        oracle_actions = []
        orientation = self.get_orientation()
        for i in range(len(shortest_path) - 1):
            prev_node = shortest_path[i]
            next_node = shortest_path[i+1]
            p1 = self.graph.nodes[prev_node]['point']
            p2 = self.graph.nodes[next_node]['point']
            direction = int(np.around(
                np.rad2deg(np.arctan2(p2[2] - p1[2], p2[0] - p1[0])))) % 360
            if direction == orientation:
                pass
            elif (direction - orientation) % 360 == 270:
                orientation = (orientation - 90) % 360
                oracle_actions.append(HabitatSimActions.TURN_LEFT)
            elif (direction - orientation) % 360 == 90:
                orientation = (orientation + 90) % 360
                oracle_actions.append(HabitatSimActions.TURN_RIGHT)
            elif (direction - orientation) % 360 == 180:
                orientation = (orientation - 180) % 360
                oracle_actions.append(HabitatSimActions.TURN_RIGHT)
                oracle_actions.append(HabitatSimActions.TURN_RIGHT)
            oracle_actions.append(HabitatSimActions.MOVE_FORWARD)
        oracle_actions.append(HabitatSimActions.STOP)
        return oracle_actions

    def get_oracle_action(self):
        return self._oracle_actions[self._episode_step_count]

    @property
    def previous_step_collided(self):
        return self._previous_step_collided

    # def find_nearest_graph_node(self, target_pos):
    #     from scipy.spatial import cKDTree
    #     all_points = np.array(
    #         [self.graph.nodes()[node]['point'] for node in self.graph.nodes()])
    #     kd_tree = cKDTree(all_points[:, [0, 2]])
    #     d, ind = kd_tree.query(target_pos[[0, 2]])
    #     return all_points[ind]

    def seed(self, seed):
        self._sim.seed(seed)

    def get_observations_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(
                position, rotation, reset_sensors=False
            )

        if success:
            sim_obs = self._sim.get_sensor_observations()

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
            return observations
        else:
            return None


@registry.register_simulator()
class DwpSimVLNCE(Simulator, ABC):
    r"""Simulator wrapper over habitat-sim
    habitat-sim repo: https://github.com/facebookresearch/habitat-sim
    Args:
        config: configuration for initializing the simulator.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        agent_config = self._get_agent_config()

        sim_sensors = []
        for sensor_name in agent_config.SENSORS:
            sensor_cfg = getattr(self.config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            sim_sensors.append(sensor_type(sensor_cfg))
            
        self._is_episode_active = None
        self._sensor_suite = SensorSuite(sim_sensors)
        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._current_scene = self.sim_config.sim_cfg.scene.id
        self._sim = habitat_sim.Simulator(self.sim_config)
        self._action_space = spaces.Discrete(
            len(self.sim_config.agents[0].action_space)
        )
        self._egomap_cache = defaultdict(dict)
        self._prev_sim_obs = None

    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        overwrite_config(
            config_from=self.config.HABITAT_SIM_V0, config_to=sim_config
        )
        sim_config.scene.id = self.config.SCENE
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=self._get_agent_config(), config_to=agent_config
        )

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            sim_sensor_cfg = habitat_sim.SensorSpec()
            overwrite_config(
                config_from=sensor.config, config_to=sim_sensor_cfg
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(
                sensor.observation_space.shape[:2]
            )
            sim_sensor_cfg.parameters["hfov"] = str(sensor.config.HFOV)

            # TODO(maksymets): Add configure method to Sensor API to avoid
            # accessing child attributes through parent interface
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type  # type: ignore
            sim_sensor_cfg.gpu2gpu_transfer = (
                self.config.HABITAT_SIM_V0.GPU_GPU
            )
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = registry.get_action_space_configuration(
            self.config.ACTION_SPACE_CONFIG
        )(self.config).get()

        return habitat_sim.Configuration(sim_config, [agent_config])

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    @property
    def action_space(self) -> Space:
        return self._action_space

    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, _ in enumerate(self.config.AGENTS):
            agent_cfg = self._get_agent_config(agent_id)
            if agent_cfg.IS_SET_START_STATE:
                self.set_agent_state(
                    agent_cfg.START_POSITION,
                    agent_cfg.START_ROTATION,
                    agent_id,
                )
                is_updated = True

        return is_updated

    def reset(self):
        sim_obs = self._sim.reset()
        if self._update_agents_state():
            sim_obs = self._sim.get_sensor_observations()
        self._is_episode_active = True
        self._prev_sim_obs = sim_obs
        return self._sensor_suite.get_observations(sim_obs)

    def step(self, action):
        sim_obs = self._sim.step(action)
        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)
        return observations
    # def step(self, action, only_allowed=True):
    #     """
    #     All angle calculations in this function is w.r.t habitat coordinate 
    #     frame, on X-Z plane where +Y is upward, -Z is forward and +X is rightward.
    #     Angle 0 corresponds to +X, angle 90 corresponds to +y and 290 corresponds 
    #     to 270.

    #     :param action: action to be taken
    #     :param only_allowed: if true, then can't step anywhere except allowed 
    #     locations
    #     :return:
    #     Dict of observations
    #     """
    #     assert self._is_episode_active, (
    #         "episode is not active, environment not RESET or "
    #         "STOP action called previously"
    #     )

    #     self._previous_step_collided = False
    #     # STOP: 0, FORWARD: 1, LEFT: 2, RIGHT: 2
    #     if action == HabitatSimActions.STOP:
    #         self._is_episode_active = False
    #     else:
    #         prev_position_index = self._receiver_position_index
    #         prev_rotation_angle = self._rotation_angle
    #         if action == HabitatSimActions.MOVE_FORWARD:
    #             # the agent initially faces -Z by default
    #             self._previous_step_collided = True
    #             for neighbor in self.graph[self._receiver_position_index]:
    #                 p1 = self.graph.nodes[self._receiver_position_index]['point']
    #                 p2 = self.graph.nodes[neighbor]['point']
    #                 direction = int(np.around(
    #                     np.rad2deg(np.arctan2(p2[2] - p1[2], p2[0] - p1[0])))) % 360
    #                 if direction == self.get_orientation():
    #                     self._receiver_position_index = neighbor
    #                     self._previous_step_collided = False
    #                     break
    #         elif action == HabitatSimActions.TURN_LEFT:
    #             # agent rotates counterclockwise, so turning left means increasing rotation angle by 90
    #             self._rotation_angle = (self._rotation_angle + 90) % 360
    #         elif action == HabitatSimActions.TURN_RIGHT:
    #             self._rotation_angle = (self._rotation_angle - 90) % 360
                
    #         self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
    #                              quat_from_angle_axis(np.deg2rad(self._rotation_angle), np.array([0, 1, 0])))
    #     self._episode_step_count += 1

    #     # log debugging info
    #     logging.debug('After taking action {}, s,r: {}, {}, orientation: {}, location: {}'.format(
    #         action, self._source_position_index, self._receiver_position_index,
    #         self.get_orientation(), self.graph.nodes[self._receiver_position_index]['point']))

    #     sim_obs = self._get_sim_observation()
    
    #     self._prev_sim_obs = sim_obs
    #     observations = self._sensor_suite.get_observations(sim_obs)

    #     return observations

    def render(self, mode: str = "rgb") -> Any:
        r"""
        Args:
            mode: sensor whose observation is used for returning the frame,
                eg: "rgb", "depth", "semantic"

        Returns:
            rendered frame according to the mode
        """
        sim_obs = self._sim.get_sensor_observations()
        observations = self._sensor_suite.get_observations(sim_obs)

        output = observations.get(mode)
        assert output is not None, "mode {} sensor is not active".format(mode)
        if not isinstance(output, np.ndarray):
            # If it is not a numpy array, it is a torch tensor
            # The function expects the result to be a numpy array
            output = output.to("cpu").numpy()

        return output

    def seed(self, seed):
        self._sim.seed(seed)

    def reconfigure(self, config: Config) -> None:
        # TODO(maksymets): Switch to Habitat-Sim more efficient caching
        is_same_scene = config.SCENE == self._current_scene
        self.config = config
        self.sim_config = self.create_sim_config(self._sensor_suite)
        if not is_same_scene:
            self._current_scene = config.SCENE
            self._sim.close()
            del self._sim
            self._sim = habitat_sim.Simulator(self.sim_config)

        self._update_agents_state()

    def geodesic_distance(
        self, position_a, position_b, episode: Optional[Episode] = None
    ):
        if episode is None or episode._shortest_path_cache is None:
            path = habitat_sim.MultiGoalShortestPath()
            if isinstance(position_b[0], List) or isinstance(
                position_b[0], np.ndarray
            ):
                path.requested_ends = np.array(position_b, dtype=np.float32)
            else:
                path.requested_ends = np.array(
                    [np.array(position_b, dtype=np.float32)]
                )
        else:
            path = episode._shortest_path_cache

        path.requested_start = np.array(position_a, dtype=np.float32)

        self._sim.pathfinder.find_path(path)

        if episode is not None:
            episode._shortest_path_cache = path

        return path.geodesic_distance

    def action_space_shortest_path(
        self, source: AgentState, targets: List[AgentState], agent_id: int = 0
    ) -> List[ShortestPathPoint]:
        r"""
        Returns:
            List of agent states and actions along the shortest path from
            source to the nearest target (both included). If one of the
            target(s) is identical to the source, a list containing only
            one node with the identical agent state is returned. Returns
            an empty list in case none of the targets are reachable from
            the source. For the last item in the returned list the action
            will be None.
        """
        raise NotImplementedError(
            "This function is no longer implemented. Please use the greedy "
            "follower instead"
        )

    @property
    def up_vector(self):
        return np.array([0.0, 1.0, 0.0])

    @property
    def forward_vector(self):
        return -np.array([0.0, 0.0, 1.0])

    def get_straight_shortest_path_points(self, position_a, position_b):
        path = habitat_sim.ShortestPath()
        path.requested_start = position_a
        path.requested_end = position_b
        self._sim.pathfinder.find_path(path)
        return path.points

    def sample_navigable_point(self):
        return self._sim.pathfinder.get_random_navigable_point().tolist()

    def is_navigable(self, point: List[float]):
        return self._sim.pathfinder.is_navigable(point)

    def semantic_annotations(self):
        r"""
        Returns:
            SemanticScene which is a three level hierarchy of semantic
            annotations for the current scene. Specifically this method
            returns a SemanticScene which contains a list of SemanticLevel's
            where each SemanticLevel contains a list of SemanticRegion's where
            each SemanticRegion contains a list of SemanticObject's.

            SemanticScene has attributes: aabb(axis-aligned bounding box) which
            has attributes aabb.center and aabb.sizes which are 3d vectors,
            categories, levels, objects, regions.

            SemanticLevel has attributes: id, aabb, objects and regions.

            SemanticRegion has attributes: id, level, aabb, category (to get
            name of category use category.name()) and objects.

            SemanticObject has attributes: id, region, aabb, obb (oriented
            bounding box) and category.

            SemanticScene contains List[SemanticLevels]
            SemanticLevel contains List[SemanticRegion]
            SemanticRegion contains List[SemanticObject]

            Example to loop through in a hierarchical fashion:
            for level in semantic_scene.levels:
                for region in level.regions:
                    for obj in region.objects:
        """
        return self._sim.semantic_scene

    def close(self):
        self._sim.close()

    def _get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.config.DEFAULT_AGENT_ID
        agent_name = self.config.AGENTS[agent_id]
        agent_config = getattr(self.config, agent_name)
        return agent_config

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        assert agent_id == 0, "No support of multi agent in {} yet.".format(
            self.__class__.__name__
        )
        return self._sim.get_agent(agent_id).get_state()

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        r"""Sets agent state similar to initialize_agent, but without agents
        creation. On failure to place the agent in the proper position, it is
        moved back to its previous pose.

        Args:
            position: list containing 3 entries for (x, y, z).
            rotation: list with 4 entries for (x, y, z, w) elements of unit
                quaternion (versor) representing agent 3D orientation,
                (https://en.wikipedia.org/wiki/Versor)
            agent_id: int identification of agent from multiagent setup.
            reset_sensors: bool for if sensor changes (e.g. tilt) should be
                reset).

        Returns:
            True if the set was successful else moves the agent back to its
            original pose and returns false.
        """
        agent = self._sim.get_agent(agent_id)
        original_state = self.get_agent_state(agent_id)
        new_state = self.get_agent_state(agent_id)
        new_state.position = position
        new_state.rotation = rotation

        # NB: The agent state also contains the sensor states in _absolute_
        # coordinates. In order to set the agent's body to a specific
        # location and have the sensors follow, we must not provide any
        # state for the sensors. This will cause them to follow the agent's
        # body
        new_state.sensor_states = dict()
        agent.set_state(new_state, reset_sensors)
        return True

    def get_observations_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(
                position, rotation, reset_sensors=False
            )

        if success:
            sim_obs = self._sim.get_sensor_observations()

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
            return observations
        else:
            return None

    def distance_to_closest_obstacle(self, position, max_search_radius=2.0):
        return self._sim.pathfinder.distance_to_closest_obstacle(
            position, max_search_radius
        )

    def island_radius(self, position):
        return self._sim.pathfinder.island_radius(position)

    def get_egomap_observation(self):
        joint_index = (self._receiver_position_index, self._rotation_angle)
        if joint_index in self._egomap_cache[self._current_scene]:
            return self._egomap_cache[self._current_scene][joint_index]
        else:
            return None
    
    def cache_egomap_observation(self, egomap):
        self._egomap_cache[self._current_scene][(
            self._receiver_position_index, self._rotation_angle)] = egomap

    @property
    def previous_step_collided(self):
        r"""Whether or not the previous step resulted in a collision

        Returns:
            bool: True if the previous step resulted in a collision, false otherwise

        Warning:
            This feild is only updated when :meth:`step`, :meth:`reset`, or :meth:`get_observations_at` are
            called.  It does not update when the agent is moved to a new loction.  Furthermore, it
            will _always_ be false after :meth:`reset` or :meth:`get_observations_at` as neither of those
            result in an action (step) being taken.
        """
        return self._prev_sim_obs.get("collided", False)