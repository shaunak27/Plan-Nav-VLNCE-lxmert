# -----------------------------------------------------------------------------#
# @authors                                                                     #
# @date     march 23, 2021                                                     #
# @brief    test candidate proposal model                                      #
# -----------------------------------------------------------------------------#
import argparse
import habitat
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch

from habitat import logger
from habitat.sims.habitat_simulator.actions import HabitatSimActions

sys.path.append("..")
from feature_extractor import get_images

from vlnce_baselines.common.environments import SimpleRLEnv
from vlnce_baselines.candidate_proposal.candidate_proposal import CandidateProposal
from vlnce_baselines.encoders.feature_extractor import FeatureExtractor

np.set_printoptions(suppress=True)

def run_exp(exp_config: str):

    # setup
    config = habitat.get_config(config_paths=exp_config)
    logger.add_filehandler(config.LOG_FILE)
    logger.info(config)
    
    # feature extractor model
    fe_config = config.FEATURE_EXTRACTOR
    fe = FeatureExtractor(fe_config)
    
    # candidate proposal model 
    cp_config = config.CANDIDATE_PROPOSAL
    cp = CandidateProposal(cp_config)

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
            
            # extract features 
            feats = fe.extract_vln_feat_caffe(img_list=rgb_imgs)
            feats = torch.from_numpy(feats).to(device=cp.get_device())
            logger.info(f"features shape: {feats.shape}")
            
            # get candidates
            feats_with_cands, candidate_hr = cp.propose_candidates(
                features=feats, 
                depth_images=depth_imgs
            )
            logger.info(f"features + candidates shape: {feats_with_cands.shape}")
            logger.info(f"candidate info: {candidate_hr}")
            
            # to verify, print out data
            scan = cp.get_scan()
            logger.info(f"scan shape: {scan.shape}")
            
            occ_map = cp.get_ocupancy_map()
            plt.imshow(occ_map, origin='lower')
            plt.savefig(os.path.join(episode_path, "occupancy_map.png"))
            plt.close()
            
            nms_pred = cp.get_nms_pred()
            plt.imshow(nms_pred, origin='lower')
            plt.savefig(os.path.join(episode_path, "waypoint_pred.png"))
            plt.close()
            
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
