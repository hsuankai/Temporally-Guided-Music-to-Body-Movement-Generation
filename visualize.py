#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:22:58 2019

@author: gaussian
"""


import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess as sp

from common.camera import *
from common.custom_dataset import CustomDataset
from common.visualization import get_resolution, get_fps, read_video


def render_animation(p, fps, output_path, azim, prediction, ground_truth=None, knee=False):
#    prediction_path = "URMP_keypoints_3D/39_1.npy"
#    orientation = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088], dtype="float32")
#    rot = orientation
    bitrate = 3000
    azim = azim
    output = output_path
    limit = len(prediction)
    size = 6
    input_video_skip = 0
    knee = knee
    fps = 30
    
#    ground_truth = True
    
    if not knee:
        parents = [[0,1], [0,2], [0,3],
                   [3,4], [4,5], [5,6],
                   [4,7], [7,8], [8,9],
                   [4,10], [10,11], [11,12]]
        joints_right = [1, 10, 11, 12]
        
#        parents = [[0,1],[1,2]]
#        joints_right = [0,1,2]
    else:
        parents = [[0,1], [0,3], [0,5],
                   [1,2], [3,4],
                   [5,6], [6,7], [7,8],
                   [6,9], [9,10], [10,11],
                   [6,12], [12,13], [13,14]]
        joints_right = [1, 2, 12, 13, 14]
    

    prediction[:, :, 2] += 0.3
    if ground_truth is not None: 
        ground_truth[:, :, 2] += 0.3
        poses = {'Prediction': prediction,
                 'Ground_truth': ground_truth}
    else:
        poses = {'Prediction': prediction}


    plt.ioff()
    fig = plt.figure(figsize=(size*len(poses), size))
    
    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, len(poses), index + 1, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius/2, radius/2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius/2, radius/2])
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title) #, pad=35
        ax.grid(False)
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    poses = list(poses.values())
    
    ## Decode video
    
    # Load video using ffmpeg
#    all_frames = []
#    for f in read_video(input_video_path, skip=input_video_skip, limit=limit):
#        all_frames.append(f)
#    if prediction.shape[0] != len(all_frames):
#        print("some frames missing.")
#    effective_length = min(prediction.shape[0], len(all_frames))
#    all_frames = all_frames[:effective_length]
    
    # start from frame 0
    for idx in range(len(poses)):
        poses[idx] = poses[idx][input_video_skip:]
    
#    if fps is None:
#        fps = get_fps(input_video_path)
    
    
    initialized = False
    image = None
    lines = []
    points = None
    
#    if limit < 1:
#        limit = len(all_frames)
#    else:
#        limit = min(limit, len(all_frames))
    
    
    def update_video(i):
        
        nonlocal initialized, image, lines, points
        
#        for n, ax in enumerate(ax_3d):
#            ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
#            ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])

        if not initialized:
            
            for j,joints in enumerate(parents):
                j_parent0, j_parent1 = joints[0], joints[1]

    
                col = 'red' if j_parent0 and j_parent1 in joints_right else 'black'
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j_parent0, 0], pos[j_parent1, 0]],
                                               [pos[j_parent0, 1], pos[j_parent1, 1]],
                                               [pos[j_parent0, 2], pos[j_parent1, 2]], zdir='z', c=col))
    

    
            initialized = True
        else:

    
            for j,joints in enumerate(parents):
                j_parent0, j_parent1 = joints[0], joints[1]
    
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j][0].set_xdata([pos[j_parent0, 0], pos[j_parent1, 0]])
                    lines_3d[n][j][0].set_ydata([pos[j_parent0, 1], pos[j_parent1, 1]])
                    lines_3d[n][j][0].set_3d_properties([pos[j_parent0, 2], pos[j_parent1, 2]], zdir='z')
    
        if i%100==0:
            print('{}/{}\n'.format(i, limit))
        
    
    fig.tight_layout()
    
    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000/fps, repeat=False)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()
#render_animation()