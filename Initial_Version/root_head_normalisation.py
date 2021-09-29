"""
Python Script to Normalise the Human 3.6m Annotations Files between -1 and 1 relative to the root (pelvic) joint.
"""

import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler

train_path = r'datasets/ECCV18_Challenge/Train/POSE/'
test_path = r'datasets/ECCV18_Challenge/Val/POSE/'
save_train_path = 'datasets/ECCV18_Challenge/Train/NORM_POSE/'
save_test_path = 'datasets/ECCV18_Challenge/Val/NORM_POSE/'

train_df = []
train_df_3d = []
train_json_file = []
for pose_csv in os.listdir(train_path):
    pose = np.genfromtxt(train_path + pose_csv, delimiter=',')
    train_json_file.append(pose.tolist())
    root_head_distance = np.hypot(pose[9, 0], pose[9, 1])
    norm_pose = pose[:, :2]/root_head_distance * 0.1
    train_df.append(norm_pose[1:].tolist())

with open(save_train_path + 'ground_truth_3d_train_poses.json', 'w') as outfile:
    json.dump(train_json_file, outfile)

with open(save_train_path + 'normalised_2d_train_poses.json', 'w') as outfile:
    json.dump(train_df, outfile)

print('Train Normalisation Done')

test_df = []
test_df_3d = []
test_json_file = []
for pose_csv in os.listdir(test_path):
    pose = np.genfromtxt(test_path + pose_csv, delimiter=',')
    test_json_file.append(pose.tolist())
    root_head_distance = np.hypot(pose[9, 0], pose[9, 1])
    norm_pose = pose[:, :2]/root_head_distance * 0.1
    test_df.append(norm_pose[1:].tolist())

with open(save_test_path + 'ground_truth_3d_test_poses.json', 'w') as outfile:
    json.dump(test_json_file, outfile)

with open(save_test_path + 'normalised_2d_test_poses.json', 'w') as outfile:
    json.dump(test_df, outfile)


print('Test Normalisation Done')