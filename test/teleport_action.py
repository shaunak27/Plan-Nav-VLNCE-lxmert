#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import numpy as np
import pytest
sys.path.append("..")
import habitat
#from habitat.utils.test_utils import sample_non_stop_action
from vlnce_baselines.common.environments import SimpleRLEnv

CFG_TEST = "teleport_action.yaml"
TELEPORT_POSITION = np.array([4.2890449, -0.15067159, 0.124366])
TELEPORT_ROTATION = np.array([0.2, -0.3, 0, 0])


def test_task_actions():
    config = habitat.get_config(config_paths=CFG_TEST)
    config.defrost()
    config.TASK.POSSIBLE_ACTIONS = config.TASK.POSSIBLE_ACTIONS + ["TELEPORT"]
    config.freeze()
    with SimpleRLEnv(config=config) as env:
        
        env.reset()
        agent_state = env.habitat_env.sim.get_agent_state()
        action = {
            "action": "TELEPORT",
            "action_args": {
                "position": TELEPORT_POSITION ,
                "rotation": TELEPORT_ROTATION,
            },
        }
        assert env.action_space.contains(action)
        env.step(action)
        agent_state = env.habitat_env.sim.get_agent_state()
        
        assert np.allclose(
            np.array(TELEPORT_POSITION, dtype=np.float32), agent_state.position
        ), "mismatch in position after teleport"
        assert np.allclose(
            np.array(TELEPORT_ROTATION, dtype=np.float32),
            np.array([*agent_state.rotation.imag, agent_state.rotation.real]),
        ), "mismatch in rotation after teleport"
        print("\nTest passed successfully\n")

if __name__ == "__main__":
    test_task_actions()