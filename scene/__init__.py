###
# Copyright (C) 2023, Computer Vision Lab, Seoul National University, https://cv.snu.ac.kr
# For permission requests, please contact robot0321@snu.ac.kr, esw0116@snu.ac.kr, namhj28@gmail.com, jarin.lee@gmail.com.
# All rights reserved.
###
import os
import random

from arguments import GSParams
from utils.system import searchForMaxIteration
from scene.dataset_readers import readDataInfo,loadCamerasFromRawData
from scene.gaussian_model import GaussianModel
from utils.graphics import BasicPointCloud


class Scene:
    gaussians: GaussianModel

    def __init__(self, traindata, gaussians: GaussianModel, opt: GSParams):
        self.traindata = traindata
        self.gaussians = gaussians
        
        info = readDataInfo(traindata, opt.white_background)
        random.shuffle(info.train_cameras)  # Multi-res consistent random shuffling
        self.cameras_extent = info.nerf_normalization["radius"]

        print("Loading Training Cameras")
        self.train_cameras = info.train_cameras        
        print("Loading Preset Cameras")
        self.preset_cameras = {}
        for campath in info.preset_cameras.keys():
            self.preset_cameras[campath] = info.preset_cameras[campath]

        self.gaussians.create_from_pcd(info.point_cloud, self.cameras_extent)
        self.gaussians.training_setup(opt)

    def getTrainCameras(self):
        return self.train_cameras
    
    def getPresetCameras(self, preset):
        assert preset in self.preset_cameras
        return self.preset_cameras[preset]
    
class Scene_zx:
    gaussians: GaussianModel

    def __init__(self, all_points, all_colors, rgb_dir, poses, width, height, gaussians: GaussianModel, opt: GSParams):
        self.gaussians = gaussians
        self.cameras_extent = 0.001
        self.point_cloud = BasicPointCloud(points=all_points, colors=all_colors, normals=None)

        self.train_cameras = loadCamerasFromRawData(rgb_dir, poses, width, height, white_background=opt.white_background)

        self.gaussians.create_from_pcd(self.point_cloud, self.cameras_extent)
        self.gaussians.training_setup(opt)

    def getTrainCameras(self):
        return self.train_cameras