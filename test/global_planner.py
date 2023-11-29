# -----------------------------------------------------------------------------#                    
# @date     june 12, 2021                                                      #
# @brief    test global planner                                                #
# -----------------------------------------------------------------------------#
import argparse
import habitat
import numpy as np
import os
import sys

from habitat import logger
from habitat.sims.habitat_simulator.actions import HabitatSimActions

sys.path.append("..")
from feature_extractor import get_images

from vlnce_baselines.common.environments import SimpleRLEnv
from vlnce_baselines.planners.global_planner import GlobalPlanner

np.set_printoptions(suppress=True)

def run_exp(exp_config: str):

    # setup
    config = habitat.get_config(config_paths=exp_config)
    logger.add_filehandler(config.LOG_FILE)
    logger.info(config)
    
    # global planner
    gp_config = config.GLOBAL_PLANNER
    gp = GlobalPlanner(config=gp_config) 
    
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
            episode = env.current_episode
            episode_id = str(episode.episode_id)
            episode_path = os.path.join(output_path, episode_id)
            instruction = {
                "instr_id": episode_id,
                "instruction": episode.instruction.instruction_text
            }
            n_step = 0

            if not os.path.exists(episode_path):
                os.makedirs(episode_path)
                
            # prepare observations
            metadata = {
                'outdir': episode_path,
                'episode_id': episode_id,
                'n': n_step,
                'write': True
            }
            rgb_imgs, depth_imgs = get_images(
                observations=observations, 
                metadata=metadata
            )
            
            choice, probs, waypoint = gp.run(
                instruction=instruction, 
                rgb_images=rgb_imgs, 
                depth_images=depth_imgs
            )
            logger.info(f"Agent choice: {choice} probs: {probs}")
            logger.info(f"Waypoint: {waypoint}")
            
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
