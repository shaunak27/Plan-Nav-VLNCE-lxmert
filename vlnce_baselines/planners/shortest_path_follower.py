#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import habitat_sim

from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_extensions.utils import polar_to_cartesian
from typing import Optional, Union
import warnings

def action_to_one_hot(action: int) -> np.array:
    one_hot = np.zeros(len(HabitatSimActions), dtype=np.float32)
    one_hot[action] = 1
    return one_hot


class ShortestPathFollower:
    def __init__(
        self,
        sim: HabitatSim,
        goal_radius: float,
        return_one_hot: bool = True,
        stop_on_error: bool = True,
    ):
        self._return_one_hot = return_one_hot
        self._sim = sim
        self._goal_radius = goal_radius
        self._follower = None
        self._current_scene = None
        self._stop_on_error = stop_on_error

    def _build_follower(self):
        if self._current_scene != self._sim.config.SCENE:
            self._follower = self._sim._sim.make_greedy_follower(
                0,
                self._goal_radius,
                stop_key=HabitatSimActions.STOP,
                forward_key=HabitatSimActions.MOVE_FORWARD,
                left_key=HabitatSimActions.TURN_LEFT,
                right_key=HabitatSimActions.TURN_RIGHT,
            )
            self._current_scene = self._sim.config.SCENE

    def _get_return_value(self, action) -> Union[int, np.array]:
        if self._return_one_hot:
            return action_to_one_hot(action)
        else:
            return action

    def get_next_action(
        self, goal_pos: np.array
    ) -> Optional[Union[int, np.array]]:
        """Returns the next action along the shortest path.
        """
        self._build_follower()
        try:
            next_action = self._follower.next_action_along(goal_pos)
        except habitat_sim.errors.GreedyFollowerError as e:
            if self._stop_on_error:
                next_action = HabitatSimActions.STOP
            else:
                raise e

        return self._get_return_value(next_action)

    def get_agent_state(self):
        return self._sim.get_agent_state()

    def set_agent_state(self,state: habitat_sim.AgentState):
        return self._sim.set_agent_state(state.position, state.rotation)

    @property
    def mode(self):
        warnings.warn(".mode is depricated", DeprecationWarning)
        return ""

    @mode.setter
    def mode(self, new_mode: str):
        warnings.warn(".mode is depricated", DeprecationWarning)
    
class ShortestRelativePathFollower(ShortestPathFollower):
    def compute_goal(self, goal_pos: np.array, goal_format: str = None,dimensionality=3):
        source_position = self.get_agent_state().position

        if goal_format == "POLAR":
            if dimensionality == 2:
                x, z = polar_to_cartesian(
                    goal_pos[0], goal_pos[1]
                )
                return np.array([x, z] + source_position, dtype=np.float32)
            else:
                x, z = polar_to_cartesian(
                    goal_pos[0]*np.sin(goal_pos[2]), goal_pos[1]
                )
                y = goal_pos[0]*np.cos(goal_pos[2])
                return np.array([x, y, z] + source_position, dtype=np.float32)
        
        elif goal_format == "CARTESIAN":   
            return goal_pos + source_position
        else :
            return goal_pos

    def get_next_action(
        self, goal_pos: np.array, goal_format: str = None,dimensionality = 3
    ) -> Optional[Union[int, np.array]]:
        """Returns the next action along the shortest path.
        """
        abs_goal = self.compute_goal(goal_pos, goal_format, dimensionality)
        self._build_follower()
        try:
            next_action = self._follower.next_action_along(abs_goal)
        except habitat_sim.errors.GreedyFollowerError as e:
            if self._stop_on_error:
                next_action = HabitatSimActions.STOP
            else:
                raise e

        return self._get_return_value(next_action)

    def set_agent_state(self,state: habitat_sim.AgentState):
        return self._sim.set_agent_state(state.position, state.rotation)

