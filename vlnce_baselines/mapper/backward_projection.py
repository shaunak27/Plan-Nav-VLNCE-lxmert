# ------------------------------------------------------------------------------
# @file: candidate_proposal.py
# @brief: Proposes candidate images from a given set of images. A candidate
# image represents an images where navigation is feasible.
# ------------------------------------------------------------------------------
from logging import Logger
import numpy as np
from pyquaternion import Quaternion

from habitat import logger
from habitat.config import Config
from itertools import zip_longest
from typing import Union

class BackwardProjection:
    def __init__(self, config: Config, camera: Config) -> None:
        r''' Initializes the Backward projection model.
        Args:
            -config: YAML projection configuration
            -camera: YAML with camera configuration
        Returns:
            None
        '''
        # projection config
        self.x_range = (config.x_range.min, config.x_range.max)
        self.x_res = config.x_range.resolution
        
        self.y_range = (config.y_range.min, config.y_range.max)
        self.y_res = config.y_range.resolution
        
        self.z_range = (config.z_range.min, config.z_range.max)
        self.z_res = config.z_range.resolution
        
        # top-down map config
        x0, xn = 0, int((self.x_range[1]-self.x_range[0])//self.x_res)+1
        # y0, yn = 0, int((self.y_range[1]-self.y_range[0])//self.y_res)+1
        z0, zn = 0, int((self.z_range[1]-self.z_range[0])//self.z_res)+1
        self.top_down_map_height = zn - z0
        self.top_down_map_width = xn - x0
        
        # camera config
        self.height = camera.HEIGHT
        self.width = camera.WIDTH
        self.hfov = camera.HFOV
        self.max_depth = camera.MAX_DEPTH
        self.min_depth = camera.MIN_DEPTH 
        self.rotation = np.array(camera.ORIENTATION)
        self.translation = np.array(camera.POSITION)
        self.fx = 1 / np.tan(self.hfov / 2) 
        self.fy = (self.width / self.height) * self.fx
        self.cx = 0.0
        self.cy = 0.0
        
        self.K = np.array([[self.fx,  0, self.cx, 0],
                           [ 0, self.fy, self.cy, 0],
                           [ 0,       0,       1, 0], 
                           [ 0,       0,       0, 1]], dtype=np.float)
        logger.info(f"Intrinsic camera matrix:\n{self.K}")
        
        # TODO: check if this operation is ok -- based on Rodrigues formula
        theta = np.linalg.norm(self.rotation)
        R = (np.identity(3) + np.sin(theta) * self.rotation +
            (1 - np.cos(theta)) * np.power(self.rotation, 2))
       
        self.H = np.identity(4)
        self.H[:3, :3] = R
        self.H[:3, -1] = self.translation
        
        logger.info(f"Extrinsic camera matrix:\n{self.H}")
        
        logger.info(f"Backward Projection is ready!")
    
    def occupancy_map(self, depth_image: np.array, single_layer: bool = False) -> np.array:
        r''' Computes an occupancy map from a depth image 
        Args:
            depth_image: numpy array corresponding to the depth image
            x_range: observable side space 
            y_range: valid height
            z_range: observable forward space
            res: resolution of the ocupancy map cells
        Returns:
            occupancy_map: following the paper: https://arxiv.org/pdf/2008.09622.pdf, 
            the occupancy map has two layers:
             * layer 1 encodes cells as occupied / free. An occupied cell is one
                       where a point in the y-axis is in this range (0.2, 1.5)
             * layer 2 encodes cells as explored / unexplored. An explored cell 
                       is one where a point exists in the y-axis.
        '''
        # convert depth image into point cloud
        xyz = self.depth2pointcloud(depth_image)
        
        # truncate the point cloud along x and z to the 'reliable' range. In the 
        # paper they truncate the point cloud to a 3x3m local space
        x, z = xyz[0, :], xyz[2, :]
        x_filter = np.logical_and((x >= self.x_range[0]), (x <= self.x_range[1]))
        z_filter = np.logical_and((z >= self.z_range[0]), (z <= self.z_range[1]))
        xz_filter = np.logical_and(x_filter, z_filter)
        xz_idx = np.argwhere(xz_filter).flatten()
        
        xyz_trunc = xyz[:, xz_idx]
        
        # encode free and occupied space
        x, y, z = xyz_trunc[0, :], xyz_trunc[1, :], xyz_trunc[2, :]
        
        y_occ = np.logical_and((y >= self.y_range[0]), (y <= self.y_range[1]))
        y_occ_idx = np.argwhere(y_occ).flatten()
        x_occ = ((x[y_occ_idx]-self.x_range[0])//self.x_res).astype(np.int32)
        z_occ = ((z[y_occ_idx]-self.z_range[0])//self.z_res).astype(np.int32)

        
        # convert to pixel position based on resolution 
        x_exp = ((x-self.x_range[0])//self.x_res).astype(np.int32)
        z_exp = ((z-self.z_range[0])//self.z_res).astype(np.int32)
        
        # creating the occupancy map
        if not single_layer:
            top_view_map = np.zeros(
                shape=(2, self.top_down_map_height, self.top_down_map_width), 
                dtype=np.int32)
            
            # occupied space
            top_view_map[0, z_occ, x_occ] = 1
            
            # exlpored space
            top_view_map[1, z_exp, x_exp] = 1
        else:
            top_view_map = np.zeros(
                shape=(self.top_down_map_height, self.top_down_map_width), 
                dtype=np.int32)
            
            # exlpored space
            top_view_map[-z_exp, x_exp] = 1
            
            # occupied space
            top_view_map[-z_occ, x_occ] = 2
                
        return xyz, xyz_trunc, top_view_map
    
    def depth2pointcloud(self, depth_image: np.array) -> np.array:
        r''' Receives a depth image and back-projects it into a point cloud 
        with world coordinates. 
        Args:
            depth_image: numpy array corresponding to the depth image
            camera_state: current state of the sensor
        Returns:
            pointcloud: 3D world coordinates of the projected depth map
        '''
        h, w, _ = depth_image.shape
        assert h == self.height and w == self.width, \
            f"Depth image dimensions ({h}x{w}) don't match config ({self.height}x{self.width})"
        
        # Now get an approximation for the true world coordinates -- see if they make sense
        # [-1, 1] for x and [1, -1] for y as array indexing is y-down while world is y-up
        # ref: https://aihabitat.org/docs/habitat-lab/view-transform-warp.html
        xs, ys = np.meshgrid(
            np.linspace(-1, 1, self.width), 
            np.linspace(1, -1, self.height)
        )
        
        depth = depth_image.reshape(1, self.height, self.width)
        
        xs = xs.reshape(1, self.height, self.width)
        ys = ys.reshape(1, self.height, self.width)
        
        # unproject
        # this is negating depth since camera looks along -z
        xys = np.vstack((xs * depth, ys * depth, -depth, np.ones(depth.shape)))
        xys = xys.reshape(4, -1)
        
        # use extrinsic camera parameters ?
        # H = self.K @ self.H
        # xyz = np.linalg.inv(self.H) @ xys
        xyz = np.linalg.inv(self.K) @ xys
        return xyz[:3]
