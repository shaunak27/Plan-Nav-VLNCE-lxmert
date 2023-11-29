import habitat
import numpy as np
import logging
import torch

from habitat import Config, Dataset
from habitat.tasks.utils import cartesian_to_polar

from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.utils.visualizations.utils import observations_to_image

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import NavRLEnv

from vlnce_baselines.planners.waypoint_planner import Planner
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union
from habitat.core.simulator import Observations, Simulator

#
# Classes
# ------------------------------------------------------------------------------
@baseline_registry.register_env(name="VLNCEPPORLEnv")
class VLNCEPPORLEnv(NavRLEnv):

    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._previous_action = None
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        self._success_distance = config.TASK_CONFIG.TASK.SUCCESS_DISTANCE
        self._previous_target_distance = None
        
        super().__init__(config, dataset)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0
        )

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._prev_goal_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
        ]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        # Distance reward
        distance_reward = 0.0
        if self._rl_config.DISTANCE_REWARD.COMPUTE:
            distance_reward = self.get_distance_reward()

        # Collision reward
        collision_count_reward = 0.0
        if self._rl_config.COLLISION_COUNT_REWARD.COMPUTE:
            collision_count_reward = self.get_collision_count_reward()

        collision_distance_reward = 0.0
        if self._rl_config.COLLISION_DISTANCE_REWARD.COMPUTE:
            collision_distance_reward = self.get_collision_distance_reward()

        # Success reward
        episode_success_reward = 0.0
        if self._episode_success(observations):
            episode_success_reward = self._rl_config.SUCCESS_REWARD

        reward += (
            distance_reward +
            collision_count_reward +
            collision_distance_reward +
            episode_success_reward
        )
        # print("reward: {} dist: {} heading: {} success: {}".format(
        #     reward,distance_reward,heading_reward,episode_success_reward))
        return reward

    def get_distance_reward(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = [goal.position for goal in self._env.current_episode.goals]
        distance = self._env.sim.geodesic_distance(
            current_position, target_position
        )
        reward = (self._prev_goal_distance - distance) * self._rl_config.DISTANCE_REWARD.SCALE
        self._prev_goal_distance = distance
        return reward

    def get_collision_count_reward(self):
        reward = 0.0
        collision_count_reward = self._rl_config.COLLISION_COUNT_REWARD
        collision_count = self._env.get_metrics()[
            collision_count_reward.MEASURE
        ]
        if collision_count > collision_count_reward.THRESH:
            reward = collision_count_reward.REWARD
        return reward

    def get_collision_distance_reward(self):
        reward = 0.0
        collision_distance_reward = self._rl_config.COLLISION_DISTANCE_REWARD
        collision_distance = self._env.get_metrics()[
            collision_distance_reward.MEASURE
        ]
        if collision_distance < collision_distance_reward.THRESH:
            reward = collision_distance_reward.REWARD
        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(
            current_position, target_position)
        return distance

    def _episode_success(self, observations):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        episode_success = (
            self._env.task.is_stop_called
            and self._distance_target() < self._success_distance
        )
        return self._env.episode_over or episode_success

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()
        return info
    
@baseline_registry.register_env(name="RelativePPORLEnv")
class RelativePPORLEnv(NavRLEnv):

    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._previous_action = None
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        self._success_distance = config.TASK_CONFIG.TASK.SUCCESS_DISTANCE

        self._forward_step = self._rl_config.FORWARD_STEP_SIZE
        self._turn_step = self._rl_config.TURN_ANGLE_SIZE * np.pi / 180.0

        self._prev_measure = {
            "distance_to_goal": 0.0,
            "heading": 0.0
        }
        self._waypoint_info = {
            'rel_heading': 0.0,
            'rel_distance': 0.0,
            'x': 0.0,
            'y': 0.0
        }
        self._agent_info = {
            'x': 0.0,
            'y': 0.0,
            'heading': 0.0
        }
        super().__init__(config, dataset)

    def get_reward_range(self):
        return (self._rl_config.SLACK_REWARD - 1.0,
                self._rl_config.SUCCESS_REWARD + 1.0)

    def reset(self):
        self._previous_action = None
        observations = super().reset()

        goal = observations["pointgoal"]
        self._waypoint_info['rel_heading'] = goal[1]
        self._waypoint_info['rel_distance'] = goal[0]
        self._waypoint_info['x'] = goal[0] * np.cos(goal[1])
        self._waypoint_info['y'] = goal[1] * np.sin(goal[1])

        self._agent_info['x'] = 0.0
        self._agent_info['y'] = 0.0
        self._agent_info['heading'] = 0.0

        self._prev_measure["distance_to_goal"] = goal[0]
        self._prev_measure["heading"] = goal[1]
        # print(self._waypoint_info)

        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        # Distance reward
        distance_reward = 0.0
        if self._rl_config.DISTANCE_REWARD.COMPUTE:
            distance_reward = self.get_distance_reward()

        # Collision reward
        collision_count_reward = 0.0
        if self._rl_config.COLLISION_COUNT_REWARD.COMPUTE:
            collision_count_reward = self.get_collision_count_reward()

        collision_distance_reward = 0.0
        if self._rl_config.COLLISION_DISTANCE_REWARD.COMPUTE:
            collision_distance_reward = self.get_collision_distance_reward()

        # Success reward
        episode_success_reward = 0.0
        if self._episode_success(observations):
            episode_success_reward = self._rl_config.SUCCESS_REWARD

        reward += (
            distance_reward +
            collision_count_reward +
            collision_distance_reward +
            episode_success_reward
        )
        # print("reward: {} dist: {} heading: {} success: {}".format(
        #     reward,distance_reward,heading_reward,episode_success_reward))
        return reward

    def get_distance_reward(self):

        # Estimate agent progress from previous action
        if self._previous_action == 1: # forward
            heading = self._agent_info['heading']
            x = self._forward_step * np.cos(heading)
            y = self._forward_step * np.sin(heading)
            self._agent_info['x'] += x
            self._agent_info['y'] += y
        elif self._previous_action == 2: # turn left
            self._agent_info['heading'] -= self._turn_step
        elif self._previous_action == 3: # turn right
            self._agent_info['heading'] += self._turn_step

        curr_metric = np.sqrt(
            (self._agent_info['x']-self._waypoint_info['x']) ** 2 +
            (self._agent_info['y']-self._waypoint_info['y']) ** 2
        )

        # curr_metric = self._env.get_metrics()["distance_to_goal"]
        prev_metric = self._prev_measure.get("distance_to_goal")
        reward = prev_metric - curr_metric
        self._prev_measure["distance_to_goal"] = curr_metric

        return reward

    def get_collision_count_reward(self):
        reward = 0.0
        collision_count_reward = self._rl_config.COLLISION_COUNT_REWARD
        collision_count = self._env.get_metrics()[
            collision_count_reward.MEASURE
        ]
        if collision_count > collision_count_reward.THRESH:
            reward = collision_count_reward.REWARD
        return reward

    def get_collision_distance_reward(self):
        reward = 0.0
        collision_distance_reward = self._rl_config.COLLISION_DISTANCE_REWARD
        collision_distance = self._env.get_metrics()[
            collision_distance_reward.MEASURE
        ]
        if collision_distance < collision_distance_reward.THRESH:
            reward = collision_distance_reward.REWARD
        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(
            current_position, target_position)
        return distance

    def _episode_success(self, observations):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        episode_success = (
            self._env.task.is_stop_called
            and self._distance_target() < self._success_distance
        )
        return self._env.episode_over or episode_success

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()
        return info


@baseline_registry.register_env(name="LocalNavigatorRLEnv")
class LocalNavigatorRLEnv(NavRLEnv):

    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._previous_action = None
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        self._success_distance = config.TASK_CONFIG.TASK.SUCCESS_DISTANCE

        self._forward_step = self._rl_config.FORWARD_STEP_SIZE
        self._turn_step = self._rl_config.TURN_ANGLE_SIZE * np.pi / 180.0

        self._prev_measure = {
            "distance_to_goal": 0.0,
            "heading": 0.0
        }
        self._waypoint_info = {
            'rel_heading': 0.0,
            'rel_distance': 0.0,
            'rel_elevation': 0.0,
            'x': 0.0,
            'y': 0.0
        }
        self._agent_info = {
            'x': 0.0,
            'y': 0.0,
            'heading': 0.0
        }
        super().__init__(config, dataset)

    def get_reward_range(self):
        return (self._rl_config.SLACK_REWARD - 1.0,
                self._rl_config.SUCCESS_REWARD + 1.0)

    def reset(self):
        self._previous_action = None
        observations = super().reset()

        goal = self._env.current_episode.goals[0]
        self._waypoint_info['rel_heading'] = goal.rel_heading
        self._waypoint_info['rel_distance'] = goal.rel_distance
        self._waypoint_info['rel_elevation'] = goal.rel_elevation
        self._waypoint_info['x'] = goal.rel_distance * np.cos(goal.rel_heading)
        self._waypoint_info['y'] = goal.rel_distance * np.sin(goal.rel_heading)

        self._agent_info['x'] = 0.0
        self._agent_info['y'] = 0.0
        self._agent_info['heading'] = 0.0

        self._prev_measure["distance_to_goal"] = goal.rel_distance
        self._prev_measure["heading"] = goal.rel_heading
        # print(self._waypoint_info)

        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        # Distance reward
        distance_reward = 0.0
        if self._rl_config.DISTANCE_REWARD.COMPUTE:
            distance_reward = self.get_distance_reward()

        # Collision reward
        collision_count_reward = 0.0
        if self._rl_config.COLLISION_COUNT_REWARD.COMPUTE:
            collision_count_reward = self.get_collision_count_reward()

        collision_distance_reward = 0.0
        if self._rl_config.COLLISION_DISTANCE_REWARD.COMPUTE:
            collision_distance_reward = self.get_collision_distance_reward()

        # Success reward
        episode_success_reward = 0.0
        if self._episode_success(observations):
            episode_success_reward = self._rl_config.SUCCESS_REWARD

        reward += (
            distance_reward +
            collision_count_reward +
            collision_distance_reward +
            episode_success_reward
        )
        # print("reward: {} dist: {} heading: {} success: {}".format(
        #     reward,distance_reward,heading_reward,episode_success_reward))
        return reward

    def get_distance_reward(self):

        # Estimate agent progress from previous action
        if self._previous_action == 1: # forward
            heading = self._agent_info['heading']
            x = self._forward_step * np.cos(heading)
            y = self._forward_step * np.sin(heading)
            self._agent_info['x'] += x
            self._agent_info['y'] += y
        elif self._previous_action == 2: # turn left
            self._agent_info['heading'] -= self._turn_step
        elif self._previous_action == 3: # turn right
            self._agent_info['heading'] += self._turn_step

        curr_metric = np.sqrt(
            (self._agent_info['x']-self._waypoint_info['x']) ** 2 +
            (self._agent_info['y']-self._waypoint_info['y']) ** 2
        )

        # curr_metric = self._env.get_metrics()["distance_to_goal"]
        prev_metric = self._prev_measure.get("distance_to_goal")
        reward = prev_metric - curr_metric
        self._prev_measure["distance_to_goal"] = curr_metric

        return reward

    def get_collision_count_reward(self):
        reward = 0.0
        collision_count_reward = self._rl_config.COLLISION_COUNT_REWARD
        collision_count = self._env.get_metrics()[
            collision_count_reward.MEASURE
        ]
        if collision_count > collision_count_reward.THRESH:
            reward = collision_count_reward.REWARD
        return reward

    def get_collision_distance_reward(self):
        reward = 0.0
        collision_distance_reward = self._rl_config.COLLISION_DISTANCE_REWARD
        collision_distance = self._env.get_metrics()[
            collision_distance_reward.MEASURE
        ]
        if collision_distance < collision_distance_reward.THRESH:
            reward = collision_distance_reward.REWARD
        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(
            current_position, target_position)
        return distance

    def _episode_success(self, observations):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        episode_success = (
            self._env.task.is_stop_called
            and self._distance_target() < self._success_distance
        )
        return self._env.episode_over or episode_success

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()
        return info

@baseline_registry.register_env(name="VLNCEDaggerSpeakerEnv")
class VLNCEDaggerSpeakerEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._success_distance = config.TASK_CONFIG.TASK.SUCCESS_DISTANCE
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self):
        # We don't use a reward for DAgger, but the baseline_registry requires
        # we inherit from habitat.RLEnv.
        return (0.0, 0.0)

    def get_reward(self, observations):
        return 0.0
    
    def step(self, *args, **kwargs) -> Tuple[Observations, Any, bool, dict]:
        action = kwargs["action"]
        # TODO: add to trajectory the following info:
        #   action taken
        #   corresponding waypoint
        #   global location (for verification)
        observations = self._env.step(*args, **kwargs)
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)
        # TODO: add a trajectory field to info and/or update it

        return observations, reward, done, info

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(
            current_position, target_position)
        return distance

    def get_done(self, observations):
        episode_success = (
            self._env.task.is_stop_called
            and self._distance_target() < self._success_distance
        )
        return self._env.episode_over or episode_success

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

@baseline_registry.register_env(name="VLNCEDaggerEnv")
class VLNCEDaggerEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._success_distance = config.TASK_CONFIG.TASK.SUCCESS_DISTANCE
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self):
        # We don't use a reward for DAgger, but the baseline_registry requires
        # we inherit from habitat.RLEnv.
        return (0.0, 0.0)

    def get_reward(self, observations):
        return 0.0

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(current_position, target_position)
        return distance

    def get_done(self, observations):
        episode_success = (
            self._env.task.is_stop_called
            and self._distance_target() < self._success_distance
        )
        return self._env.episode_over or episode_success

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


@baseline_registry.register_env(name="VLNCEInferenceEnv")
class VLNCEInferenceEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self):
        return (0.0, 0.0)

    def get_reward(self, observations):
        return 0.0

    def get_done(self, observations):
        return self._env.episode_over

    def get_info(self, observations):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }

class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_done(self, observations):
        return self._env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

class WpRLEnv(habitat.RLEnv):

    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._previous_action = None
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        self._success_distance = config.TASK.SUCCESS_DISTANCE

        self._forward_step = self._rl_config.FORWARD_STEP_SIZE
        self._turn_step = self._rl_config.TURN_ANGLE_SIZE * np.pi / 180.0

        self._prev_measure = {
            "distance_to_goal": 0.0,
            "heading": 0.0
        }
        self._waypoint_info = {
            'rel_heading': 0.0,
            'rel_distance': 0.0,
            'rel_elevation': 0.0,
            'x': 0.0,
            'y': 0.0
        }
        self._agent_info = {
            'x': 0.0,
            'y': 0.0,
            'heading': 0.0
        }
        super().__init__(config, dataset)

    def get_reward_range(self):
        return (self._rl_config.SLACK_REWARD - 1.0,
                self._rl_config.SUCCESS_REWARD + 1.0)

    def reset(self):
        self._previous_action = None
        observations = super().reset()

        goal = self._env.current_episode.goals[0]
        self._waypoint_info['rel_heading'] = goal.rel_heading
        self._waypoint_info['rel_distance'] = goal.rel_distance
        self._waypoint_info['rel_elevation'] = goal.rel_elevation
        self._waypoint_info['x'] = goal.rel_distance * np.cos(goal.rel_heading)
        self._waypoint_info['y'] = goal.rel_distance * np.sin(goal.rel_heading)

        self._agent_info['x'] = 0.0
        self._agent_info['y'] = 0.0
        self._agent_info['heading'] = 0.0

        self._prev_measure["distance_to_goal"] = goal.rel_distance
        self._prev_measure["heading"] = goal.rel_heading
        # print(self._waypoint_info)

        return observations

    def step(self, *args, **kwargs):
        self._previous_action = args[0]
        return super().step(*args, **kwargs)

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        # Distance reward
        distance_reward = 0.0
        if self._rl_config.DISTANCE_REWARD.COMPUTE:
            distance_reward = self.get_distance_reward()

        # Collision reward
        collision_count_reward = 0.0
        if self._rl_config.COLLISION_COUNT_REWARD.COMPUTE:
            collision_count_reward = self.get_collision_count_reward()

        collision_distance_reward = 0.0
        if self._rl_config.COLLISION_DISTANCE_REWARD.COMPUTE:
            collision_distance_reward = self.get_collision_distance_reward()

        # Success reward
        episode_success_reward = 0.0
        if self._episode_success(observations):
            episode_success_reward = self._rl_config.SUCCESS_REWARD

        reward += (
            distance_reward +
            collision_count_reward +
            collision_distance_reward +
            episode_success_reward
        )
        # print("reward: {} dist: {} heading: {} success: {}".format(
        #     reward,distance_reward,heading_reward,episode_success_reward))
        return reward

    def get_distance_reward(self):

        # Estimate agent progress from previous action
        if self._previous_action == 1: # forward
            heading = self._agent_info['heading']
            x = self._forward_step * np.cos(heading)
            y = self._forward_step * np.sin(heading)
            self._agent_info['x'] += x
            self._agent_info['y'] += y
        elif self._previous_action == 2: # turn left
            self._agent_info['heading'] -= self._turn_step
        elif self._previous_action == 3: # turn right
            self._agent_info['heading'] += self._turn_step

        curr_metric = np.sqrt(
            (self._agent_info['x']-self._waypoint_info['x']) ** 2 +
            (self._agent_info['y']-self._waypoint_info['y']) ** 2
        )

        # curr_metric = self._env.get_metrics()["distance_to_goal"]
        prev_metric = self._prev_measure.get("distance_to_goal")
        reward = prev_metric - curr_metric
        self._prev_measure["distance_to_goal"] = curr_metric

        return reward

    def get_collision_count_reward(self):
        reward = 0.0
        collision_count_reward = self._rl_config.COLLISION_COUNT_REWARD
        collision_count = self._env.get_metrics()[
            collision_count_reward.MEASURE
        ]
        if collision_count > collision_count_reward.THRESH:
            reward = collision_count_reward.REWARD
        return reward

    def get_collision_distance_reward(self):
        reward = 0.0
        collision_distance_reward = self._rl_config.COLLISION_DISTANCE_REWARD
        collision_distance = self._env.get_metrics()[
            collision_distance_reward.MEASURE
        ]
        if collision_distance < collision_distance_reward.THRESH:
            reward = collision_distance_reward.REWARD
        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(
            current_position, target_position)
        return distance

    def _episode_success(self, observations):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        episode_success = (
            self._env.task.is_stop_called
            and self._distance_target() < self._success_distance
        )
        return self._env.episode_over or episode_success

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()
        return info


@baseline_registry.register_env(name="MapNavEnv")
class MapNavEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._config = config
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG

        self._previous_target_distance = None
        self._previous_action = None
        self._previous_observation = None
        self._episode_distance_covered = None
        self._success_distance = self._core_env_config.TASK.SUCCESS_DISTANCE
        
        self._num_actions = self._config.TASK_CONFIG.TASK.ACTION_MAP.MAP_SIZE ** 2
        super().__init__(self._core_env_config, dataset)

        self.planner = Planner(
            task_config=config.TASK_CONFIG, 
            model_dir=self._config.CHECKPOINT_FOLDER, 
            masking=self._config.MASKING
        )
        torch.set_num_threads(1)

    def reset(self):
        self._previous_action = None

        observations = super().reset()
        self.planner.update_map_and_graph(observations)
        self.planner.add_maps_to_observation(observations)
        self._previous_observation = observations
        logging.debug(super().current_episode)

        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
        ]
        return observations

    def step(self, *args, **kwargs):
        intermediate_goal = kwargs["action"]
        
        self._previous_action = intermediate_goal
        goal = self.planner.get_map_coordinates(intermediate_goal)
        
        # print(f"action: {intermediate_goal} goal: {goal}")
        
        stop = int(self._num_actions // 2) == intermediate_goal
        observation = self._previous_observation
        cumulative_reward = 0
        done = False
        reaching_waypoint = False
        cant_reach_waypoint = False
        if len(self._config.VIDEO_OPTION) > 0:
            rgb_frames = list()

        for step_count in range(self._config.PREDICTION_INTERVAL):
            if step_count != 0 and not self.planner.check_navigability(goal):
                cant_reach_waypoint = True
                break
            
            action = self.planner.plan(observation, goal, stop=stop)
            
            observation, reward, done, info = super().step({"action": action})
            # TODO: fix observation size
            if len(self._config.VIDEO_OPTION) > 0:
                if "rgb" not in observation:
                    observation["rgb"] = np.zeros((self.config.DISPLAY_RESOLUTION,
                                                   self.config.DISPLAY_RESOLUTION, 3))
                frame = observations_to_image(observation, info)
                rgb_frames.append(frame)
                # audios.append(observation['audiogoal'])
            cumulative_reward += reward
            if done:
                self.planner.reset()
                observation = self.reset()
                break
            else:
                self.planner.update_map_and_graph(observation)
                # reaching intermediate goal
                x, y = self.planner.mapper.get_maps_and_agent_pose()[2:4]
                if (x - goal[0]) == (y - goal[1]) == 0:
                    reaching_waypoint = True
                    break

        if not done:
            self.planner.add_maps_to_observation(observation)
        self._previous_observation = observation
        info['reaching_waypoint'] = done or reaching_waypoint
        info['cant_reach_waypoint'] = cant_reach_waypoint
        if len(self._config.VIDEO_OPTION) > 0:
            assert len(rgb_frames) != 0
            info['rgb_frames'] = rgb_frames

        return observation, cumulative_reward, done, info

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = 0

        if self._rl_config.WITH_TIME_PENALTY:
            reward += self._rl_config.SLACK_REWARD

        if self._rl_config.WITH_DISTANCE_REWARD:
            current_target_distance = self._distance_target()
            # if current_target_distance < self._previous_target_distance:
            reward += (self._previous_target_distance - current_target_distance) * self._rl_config.DISTANCE_REWARD_SCALE
            self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD
            logging.debug('Reaching goal!')

        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = [goal.position for goal in self._env.current_episode.goals]
        distance = self._env.sim.geodesic_distance(
            current_position, target_position
        )
        return distance

    def _episode_success(self):
        if (
                self._env.task.is_stop_called
                and self._distance_target() < self._success_distance
                # and self._env.sim.reaching_goal
        ):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    # for data collection
    def get_current_episode_id(self):
        return self.habitat_env.current_episode.episode_id

    def global_to_egocentric(self, pg):
        return self.planner.mapper.global_to_egocentric(*pg)

    def egocentric_to_global(self, pg):
        return self.planner.mapper.egocentric_to_global(*pg)
