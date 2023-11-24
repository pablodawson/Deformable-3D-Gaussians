#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from gaussian_to_unity.utils import *
from gaussian_to_unity.converter import gaussian_timestep_to_unity, gaussian_static_data_to_unity
import time as tm



def save_frame(viewpoint_camera, pc, d_xyz, d_rotation, d_scaling , pipe, scaling_modifier = 1.0, 
               stage="fine", order_indexes=None, basepath = "output", 
               idx=0, args=None):

    # -- static data --
    # COLORS

    if idx==0:
        
        shs = pc.get_features
        dc = pc._features_dc

        gaussian_static_data_to_unity(pc, pc.get_xyz.shape[0], pc._scaling, pc._rotation, dc, 
                                      shs, pc._opacity, order_indexes, args= args, basepath=basepath)
    
    # Create Unity compatible frames for each gaussian state (only position at the moment)
    gaussian_timestep_to_unity(pc, d_xyz, d_scaling, d_rotation, order_indexes, debug=True, 
                               args=args, basepath=basepath, idx=idx)