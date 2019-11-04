"""
pretrained model
"""

from tensorflow import keras
import poselib

pose_approx = poselib.PoseDetector(True)
pose_correct = poselib.PoseDetector(False)

for video in videos:
    approx_poses = []
    correct_poses = []

    for frame in video:
        pa = pose_approx(frame)
        pc = pose_correct(frame)
        approx_poses.append(pa)
        correct_poses.append(pc)
