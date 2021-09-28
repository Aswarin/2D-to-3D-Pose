"""
Python Script to Normalise the Human 3.6m Annotations Files between -1 and 1 relative to the root (pelvic) joint.
"""

import numpy as np
import os
import json

train_path = r'datasets/ECCV18_Challenge/Train/POSE/'
test_path = r'datasets/ECCV18_Challenge/Val/POSE/'
save_train_path = 'datasets/ECCV18_Challenge/Train/NORM_POSE/'
save_test_path = 'datasets/ECCV18_Challenge/Val/NORM_POSE/'

train_df = []
train_df_3d = []
for pose_csv in os.listdir(train_path):
    pose = np.genfromtxt(train_path + pose_csv, delimiter=',')
    pose_max = np.max(abs(pose))
    norm_pose = pose / pose_max
    train_df_3d.append(norm_pose.tolist())
    norm_pose = norm_pose[1:, :2]
    train_df.append(norm_pose.tolist())

with open(save_train_path + 'ground_truth_3d_train_poses.json', 'w') as outfile:
    json.dump(train_df_3d, outfile)

with open(save_train_path + 'normalised_2d_train_poses.json', 'w') as outfile:
    json.dump(train_df, outfile)

print('Train Normalisation Done')

test_df = []
test_df_3d = []
for pose_csv in os.listdir(test_path):
    pose = np.genfromtxt(test_path + pose_csv, delimiter=',')
    pose_max = np.max(abs(pose))
    norm_pose = pose / pose_max
    test_df_3d.append(norm_pose.tolist())
    norm_pose = norm_pose[1:, :2]
    test_df.append(norm_pose.tolist())


with open(save_test_path + 'normalised_2d_test_poses.json', 'w') as outfile:
    json.dump(test_df, outfile)

with open(save_test_path + 'ground_truth_3d_test_poses.json', 'w') as outfile:
    json.dump(test_df_3d, outfile)

print('Test Normalisation Done')

