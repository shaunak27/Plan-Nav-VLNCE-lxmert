# -----------------------------------------------------------------------------#                                                                   #
# @date     june 12, 2021                                                      #
# @brief    test semantics                                                     #
# -----------------------------------------------------------------------------#
import argparse
import habitat
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append("..")

from habitat import logger
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from vlnce_baselines.common.environments import SimpleRLEnv
from vlnce_baselines.mapper.backward_projection import BackwardProjection

np.set_printoptions(suppress=True)
from feature_extractor import get_images

def print_region(region, pos):
    print(f"\tAgent pos: {pos}")
    print("\tRegion \tid: {} name: {} aabb-center: {} aabb-sizes: {}".format(
        region.id, 
        region.category.name(), 
        region.aabb.center, 
        region.aabb.sizes
    ))
    
def is_inside(aabb1, aabb2):
    a1 = np.array(aabb1.center - aabb1.sizes / 2)
    b1 = np.array(aabb1.center + aabb1.sizes / 2)
    
    a2 = np.array(aabb2.center - aabb2.sizes / 2)
    b2 = np.array(aabb2.center + aabb2.sizes / 2)
    
    return (
        np.all(a1 <= a2) and np.all(b1 >= a2) and 
        np.all(a1 <= b2) and np.all(b1 >= b2)
    )
        
    

def run_exp(exp_config: str):
    # setup
    config = habitat.get_config(config_paths=exp_config)
    logger.add_filehandler(config.LOG_FILE)
    logger.info(config)

    split = config.DATASET.SPLIT
    output_path = os.path.join(config.OUT_DIR, split)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with SimpleRLEnv(config=config) as env:

        for i, eps in enumerate(env.episodes):
            if (i+1) > config.NUM_EPISODES:
                break

            # reset episode
            observations = [env.reset()]
            print(f"Episode: {env.current_episode.episode_id}")
            
            state = env._env.sim.get_agent_state()
            pos = state.position
            
            # print(dir(env._env.sim.semantic_annotations()))
            regions = env._env.sim.semantic_annotations().regions 
            agent_region = None
            
            for region in regions:
                name = region.category.name()
                A = np.array(region.aabb.center - (region.aabb.sizes / 2))
                B = np.array(region.aabb.center + (region.aabb.sizes / 2))
                
                if np.all(A <= pos) and np.all(pos <= B):
                    print(f"Found region")
                    
                    if agent_region == None:
                        agent_region = region
                        print_region(agent_region, pos)
                    
                    else:
                        if is_inside(region.aabb, agent_region.aabb):
                            agent_region = region
                            print(f"Region update")
                            print_region(agent_region, pos)
            
            # if agent_region == None:
            #     print(f"Could not find region")
            #     for region in regions:
            #         print_region(region, pos)
                        
            env.step(HabitatSimActions.STOP)

        env.habitat_env.close()
        env.close()
    logger.info(f"Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    args = parser.parse_args()
    run_exp(**vars(args))


if __name__ == "__main__":
    main()
