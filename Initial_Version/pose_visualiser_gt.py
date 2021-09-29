import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import json

pose_choice = 7895

with open('datasets/ECCV18_Challenge/Val/NORM_POSE/ground_truth_3d_test_poses.json', 'r') as f:
    predicted_poses = json.load(f)

pose = np.array(predicted_poses[pose_choice])
check = pose
buff_large = np.zeros((32, 3))
buff_large[(0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27), :] = pose
pose = buff_large.transpose()

kin = np.array([[0, 12], [12, 13], [13, 14], [15, 14], [13, 17], [17, 18], [18, 19], [13, 25], [25, 26], [26, 27],
                [0, 1], [1, 2], [2, 3], [0, 6], [6, 7], [7, 8]])
order = np.array([0, 2, 1])

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()

ax = fig.gca(projection='3d')
ax.view_init(azim=-45, elev=15)

for link in kin:
    ax.plot(pose[0, link], pose[2, link], -pose[1, link], linewidth=5.0)

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Z (Model Predictions)')
ax.set_zlabel('Y')
ax.set_aspect('auto')

X = pose[0, :]
Y = pose[2, :]
Z = -pose[1, :]
max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0

mid_x = (X.max() + X.min()) * 0.5
mid_y = (Y.max() + Y.min()) * 0.5
mid_z = (Z.max() + Z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
plt.show()