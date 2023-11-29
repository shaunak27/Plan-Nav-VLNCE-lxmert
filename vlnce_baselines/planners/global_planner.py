# ------------------------------------------------------------------------------
# @file: global_planner.py
# @brief: Given visio-linguistic features computes the next waypoint.
# ------------------------------------------------------------------------------
# import base64
# import csv
# import json
from logging import log
# import math
import numpy as np
from pickle import NONE
# import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# !/usr/bin/env python

from habitat import logger
from habitat.config import Config
from numpy.lib.utils import lookfor
from typing import Dict, List, Sequence
from itertools import zip_longest
#
# import rospy
# import ros_numpy
# from sensor_msgs.msg import Image
# from std_srvs.srv import Trigger, TriggerResponse
# from vln_agent.srv import Instruction, InstructionResponse
# from vln_agent.msg import InstructionResult
#
# from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
# import actionlib
# from actionlib_msgs.msg import *
# from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, Twist

from vlnce_baselines.candidate_proposal.candidate_proposal import CandidateProposal
from vlnce_baselines.encoders.feature_extractor import FeatureExtractor
from vlnce_baselines.agents.r2r_envdrop.model import EncoderLSTM, AttnDecoderLSTM
from vlnce_baselines.agents.r2r_envdrop.utils import Tokenizer, read_vocab, angle_feature

class GlobalPlanner:
    def __init__(self, config: Config) -> None:
        r''' Initializes the Global Planner.
        Args:
            -config: YAML file containing the configuration information.
        Returns:
            None
        '''
        # Fire up some networks
        self.config = config
        self.device = (
            torch.device("cuda", self.config.gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        
        # feature extractor model 
        fe_config = self.config.FEATURE_EXTRACTOR
        self.fe = FeatureExtractor(fe_config)
    
        # candidate proposal model 
        cp_config = self.config.CANDIDATE_PROPOSAL
        self.cp = CandidateProposal(cp_config)
        
        # TODO: add local planner 
        #
        #
    
        # vln agent (global planner)
        self.vocab_path = config.vocab_file
        self.vocab = read_vocab(self.vocab_path)
        self.max_input = config.max_word_input
        self.tok = Tokenizer(vocab=self.vocab, encoding_length=self.max_input)

        # TODO today: setup vocab path, download model, and try inferencing
        self.load_model()

        # Services etc
        # service = rospy.Service('agent/instruct', Instruction, self.instruct)
        # cancel = rospy.Service('agent/instruct/cancel', Trigger, self.cancel_instruct)

        # Subscribe to features and action candidates.
        # self.feat_sub = rospy.Subscriber('subgoal/features', Image, self.process_feats)
        # self.waypoint_sub = rospy.Subscriber('subgoal/waypoints', PoseArray,
        #                                      self.process_waypoints)
        # self.feat_stamp = rospy.Time.now()

        self.instr_id = None  # Not executing an instruction
        self.image_paths = []  # Collect the file names of panos seen on the trajectory
        self.image_timestamps = []  # As well as the image timestamps
        self.max_steps = self.config.max_actions

        # Connect to theta capture service
        # theta_capture = rospy.get_param('pano_capture_service', 'theta/capture')
        # rospy.loginfo('Agent waiting for %s service' % theta_capture)
        # rospy.wait_for_service(theta_capture)
        # self.cam_service = rospy.ServiceProxy(theta_capture, Trigger)
        # rospy.loginfo('Agent connected to for %s service' % theta_capture)

        # Connect to move_base
        # rospy.loginfo("Agent waiting for move_base")
        # self.move_base = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        # self.move_base.wait_for_server()
        # rospy.loginfo("Agent connected to move_base")

        # Publisher to manually control the robot (e.g. to stop it)
        # self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        #
        # # Publish notifcation when done
        # self.pub = rospy.Publisher('agent/result', InstructionResult, queue_size=200)
        #
        # rospy.loginfo("Agent ready for instruction")
        # rospy.spin()
        self.reset()
        logger.info(f"Global planner is ready!")
        
    def reset(self) -> None:
        r''' Resets the Global Planner.
        Args:
            None
        Returns:
            None
        '''
        self.step = 0
        self.choose_stop = False
        self.candidates_hr = []
        
    def load_model(self) -> None:
        r''' Load the PyTorch instruction encoder, action decoder 
        Args:
            None
        Returns:
            None
        '''
        self.weights_file = self.config.agent_weights_file
        model_weights = torch.load(self.weights_file, map_location=self.device)

        def recover_state(name, model):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(model_weights[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(model_weights[name]['state_dict'])
            model.load_state_dict(state)

        self.padding_idx = self.config.padding_idx
        wemb = self.config.word_embedding
        dropout = self.config.dropout
        bidir = self.config.bidirectional
        rnn_dim = self.config.rnn_dim
        enc_hidden_size = rnn_dim // 2 if bidir else rnn_dim
        self.encoder = EncoderLSTM(self.tok.vocab_size(), wemb, enc_hidden_size,
                                   self.padding_idx,
                                   dropout, bidirectional=bidir).to(self.device)
        recover_state('encoder', self.encoder)
        self.encoder.eval()
        logger.info('Agent loaded encoder!')

        aemb = self.config.action_embedding
        self.cnn_feat_size = self.config.cnn_feature_size
        self.angle_feat_size = self.config.angle_feature_size
        feat_dropout = self.config.feature_dropout
        self.decoder = AttnDecoderLSTM(aemb, rnn_dim, dropout,
                                       feature_size=self.cnn_feat_size + self.angle_feat_size,
                                       angle_feat_size=self.angle_feat_size,
                                       feat_dropout=feat_dropout).to(self.device)
        recover_state('decoder', self.decoder)
        self.decoder.eval()
        logger.info('Agent loaded decoder!')
        
    def run(self, instruction, rgb_images: list, depth_images: list):
        ''' Runs the global planner.
        Args
            -instruction: text-based instruction
            -rgb_images: panorama of 36 rgb images {up, middle, down}_viewpoints 
            -depth_images: panorama of 12 depth images {middle}_viewpoints
        Returns:
            -choice: candidate index corresponding to the choice of the planner
            -probs: probability distribution of the candidate selection
            -waypoint: waypoint information of the selected candidate 
        '''
        if not self.choose_stop:
            # extract features
            rgb_feats = self.fe.extract_vln_feat_caffe(img_list=rgb_images)
            
            self.forward(
                instruction=instruction, 
                rgb_feats=rgb_feats, 
            )
            
            choice, probs =  self.choose_waypoint(depth_images)
            n_choices = len(probs)
            if choice == (n_choices - 1):
                self.choose_stop = True
        
            self.step += 1
            
        if self.choose_stop:
            logger.info(f"Agent stopped!")
            return 0, 0, {}
    
        return choice, probs, self.candidates_hr[choice]
        
    def forward(self, instruction: Dict, rgb_feats):
        r''' Runs a forward pass of the instruction and features. 
        Args:
            img_features: 36+num_candidates+1 x 2050
            (1 for stop action)
            (2050 for 2048+heading+elevation)
        Return:
            None
        '''
        # process instruction in encoder
        self.forward_instruction(instruction)
        # process feats
        self.process_feats(rgb_feats)
    
    def forward_instruction(self, req):
        r''' Process a received instruction and trigger the camera 
        Args:
            -req: text-based instruction
        Returns:
            None
        '''
        logger.info('Instr_id %s: %s' % (req["instr_id"], req["instruction"]))
        self.instr_id = req["instr_id"]
        self.step = 0
        self.instruction = req["instruction"]
        encoding = self.tok.encode_sentence(req["instruction"])
        logger.debug(
            'Agent encoded instructed as: %s' % str(self.tok.decode_sentence(encoding)))
        seq_tensor = np.array(encoding).reshape(1, -1)
        seq_lengths = list(np.argmax(seq_tensor == self.padding_idx, axis=1))
        seq_tensor = torch.from_numpy(seq_tensor[:, :seq_lengths[0]]).to(
            device=self.device).long()
        self.ctx_mask = (seq_tensor == self.padding_idx)[:, :seq_lengths[0]].to(
            device=self.device).byte()
        with torch.no_grad():
            self.ctx, self.h_t, self.c_t = self.encoder(
                seq_tensor, seq_lengths)
            self.h1 = self.h_t
        logger.info('Agent has processed encoder!')
        # return result

    def process_feats(self, img_feats):
        r''' Features to torch ready for the network 
        Args:
            -img_feats: panoramic features as numpy array
        Returns:
            None
        '''
        if self.instr_id is not None:
            # TODO: figure out what does self.features represents
            # self.feat_stamp = feat_msg.header.stamp
            # feat = ros_numpy.numpify(feat_msg)
            self.features = torch.from_numpy(img_feats).to(device=self.device)
            # self.features = self.features.transpose(0, 1)  # (36+N+1, 2050)
            logger.info('Agent received features: %s' % (str(img_feats.shape)))
        else:
            logger.info('Agent dropped features!')

    def choose_waypoint(self, depth_images):
        r''' Chooses a waypoint from depth images. 
        Args:
            -depth_images: panoramic depth image as a list of numpy arrays
        Returns:
            -choice: index of candidate
            -probs: probability distribution of candidate selection
        '''
        # predict subgoal 
        self.features, self.candidates_hr = self.cp.propose_candidates(
            features=self.features, depth_images=depth_images
        )

        # Run one step of the decoder network
        input_a_t, f_t, candidate_feat = self.get_input_feat()

        with torch.no_grad():
            self.h_t, self.c_t, logit, self.h1 = self.decoder(
                input_a_t, f_t, candidate_feat, self.h_t, self.h1, self.c_t, 
                self.ctx, self.ctx_mask
            )
            
        # Select the best action
        _, a_t = logit.max(1)
        choice = a_t.item()
        probs = F.softmax(logit, 1).squeeze().cpu().numpy()
        
        return choice, probs

    def get_input_feat(self):
        r''' Construct inputs to a decoding step of the agent 
        Args: 
            None
        Returns:
            
        '''
        # What is the agent's current heading? Decoded from first waypoint
        # TODO: remove anderson's code for now
        # r, p, y = quaternion_to_euler(self.waypoints.poses[-1].orientation)
        # agent_matt_heading = 0.5 * math.pi - y  # In matterport, pos heading turns right from y-axis
        # # snap agent heading to 30 degree increments, to match the sim
        # headingIncrement = math.pi * 2.0 / 12
        # heading_step = int(np.around(agent_matt_heading / headingIncrement))
        # if heading_step == 12:
        #     heading_step = 0
        # agent_matt_heading = heading_step * headingIncrement

        # Input action embedding, based only on current heading
        # Currently just heading set as 0
        # TODO: check if this is correct (
        #  are we using relative heading?
        #  or what heading should use?
        #  then how to represent candidate heading?)
        input_a_t = angle_feature(0, 0, self.angle_feat_size)
        input_a_t = torch.from_numpy(input_a_t).to(
            device=self.device).unsqueeze(0)

        # Image / candidate feature plus relative orientation encoding in ros coordinates
        # TODO: why subtract using 0.5 * pi? remove for now
        # feat_matt_heading = 0.5 * math.pi - self.features[:, -2]
        feat_elevation = self.features[:, -1]
        feat_rel_heading = self.features[:, -2]
        # feat_rel_heading = feat_matt_heading - agent_matt_heading
        angle_encoding = np.zeros((self.features.shape[0], self.angle_feat_size),
                                  dtype=np.float32)
        try:
            for i in range(self.features.shape[
                    0] - 1):  # Leave zeros in last position (stop vector)
                angle_encoding[i] = angle_feature(feat_rel_heading[i],
                                                  feat_elevation[i],
                                                  self.angle_feat_size)
        except:
            import pdb
            pdb.set_trace()
        angle_encoding = torch.from_numpy(
            angle_encoding).to(device=self.device)
        features = torch.cat((self.features[:, :-2], angle_encoding), dim=1).unsqueeze(
            0)
        f_t = features[:, :36]
        candidate_feat = features[:, 36:]

        return input_a_t.float(), f_t.float(), candidate_feat.float()
    
    def is_done(self) -> bool:
        return self.step >= self.max_steps or self.choose_stop

    # def move(self, goal):
    #     rospy.logdebug('Agent moving to goal')
    #     # Send the goal pose to the MoveBaseAction server
    #     self.move_base.send_goal(goal)

    #     # Allow 1 minute to get there
    #     finished_within_time = self.move_base.wait_for_result(
    #         rospy.Duration(60))

    #     # If we don't get there in time, abort the goal
    #     if not finished_within_time:
    #         self.move_base.cancel_goal()
    #         rospy.logwarn(
    #             "Agent timed out achieving goal, trigger pano anyway")
    #         # self.stop('move_base timed out')
    #         self.trigger_camera()
    #     else:
    #         # We made it! or we gave up
    #         state = self.move_base.get_state()
    #         if state == GoalStatus.SUCCEEDED:
    #             rospy.logdebug("Agent reached waypoint")
    #             self.trigger_camera()
    #         else:
    #             rospy.logwarn(
    #                 "Agent failed to reach waypoint, trigger pano anyway")
    #             # self.stop('move_base failed')
    #             self.trigger_camera()

    # def trigger_camera(self):
    #     ''' Trigger the pano camera '''
    #     result = self.cam_service()
    #     if not result.success:
    #         err = 'Could not trigger theta camera: %s' % result.message
    #         rospy.logerr(err)
    #         return InstructionResponse(success=False, message=err)
    #     self.image_paths.append(result.message)
    #     self.image_timestamps.append(rospy.Time.now())
    #     return InstructionResponse(success=True)

    # def stop(self, reason):
    #     ''' Bookkeeping for stopping an episode '''
    #     rospy.loginfo('Agent stopping due to: %s' % reason)
    #     if self.instr_id is not None:
    #         result = InstructionResult()
    #         result.header.stamp = rospy.Time.now()
    #         result.instr_id = self.instr_id
    #         result.image_filenames = self.image_paths
    #         result.image_timestamps = self.image_timestamps
    #         result.reason = reason
    #         result.start_time = self.start_time
    #         result.end_time = rospy.Time.now()
    #         self.pub.publish(result)
    #         self.instr_id = None
    #         self.image_paths = []
    #         self.image_timestamps = []

    # def cancel_instruct(self, req):
    #     ''' Cancel service callback '''
    #     if self.instr_id:
    #         # TODO can we cancel move base?
    #         self.stop('instruction_cancelled')
    #         return TriggerResponse(success=True)
    #     else:
    #         return TriggerResponse(success=False,
    #                                message='No instruction being processed')

    # def shutdown(self):
    #     # Cancel any active goals
    #     self.move_base.cancel_goal()
    #     self.stop('agent was shutdown')
    #     rospy.sleep(2)
    #     # Stop the robot
    #     self.cmd_vel_pub.publish(Twist())
    #     rospy.sleep(1)


# if __name__ == "__main__":
#     config = {
#         "vocab_file": "data/gplanner-models/train_vocab.txt",
#         "max_word_count": 80,
#         "device": "cuda:0",
#         "agent_weights_file": "data/gplanner-models/best_val_unseen.pth",
#         "padding_idx": 0,
#         "word_embedding": 256,
#         "dropout": 0.5,
#         "bidirectional": True,
#         "rnn_dim": 512,
#         "action_embedding": 64,
#         "cnn_feature_size": 2048,
#         "angle_feature_size": 128,
#         "feature_dropout": 0.4
#     }
#     navigator = GPlannerNavigator(config)
#     instr = {
#         "instr_id": 1,
#         "instruction": "go straight and turn left, stop at the floor."
#     }
#     navigator.forward(instr, None, None)
