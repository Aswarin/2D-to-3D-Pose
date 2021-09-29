import json
import numpy as np
from common import utils
from scipy.spatial import procrustes
import os

"""
This file is used to perform procrustes alignment to the predicted output. 
What this does is scale and rotate the predicted 3D pose from the generator the minimise the distance 
between this pose and the ground truth 3D pose. No manipulation is done this is purely scaling and rotation as 
the predicted 3D pose is no to scale with the ground truth 3D pose and therefore a direct comparison is not 
feasible
"""

input_size = (16, 2)  # The input to the generator and discriminator [16 (x,y) keypoints]
generator_weight_files = 'training_checkpoints/amazon_lab_paper1/gen_weights/'  # Load in the generator weights from a h5 file
ground_truth_file = 'datasets/ECCV18_Challenge/Val/NORM_POSE/ground_truth_3d_test_poses.json'
normalised_gt_file = 'datasets/ECCV18_Challenge/Val/NORM_POSE/normalised_2d_test_poses.json'

with open(normalised_gt_file) as f:
    input_val_data = np.array(json.load(f))

with open(ground_truth_file) as f:
    ground_truth_val_data = np.array(json.load(f))

best_error = 99999
best_weight = None
roots = np.zeros((len(input_val_data), 1, 3))
# Build generator and load in its weights
generator = utils.build_fc_generator(input_size)
for weight_file in os.listdir(generator_weight_files):
    generator.load_weights(generator_weight_files + weight_file)

    predictions = generator.predict(input_val_data)
    predictions = np.reshape(predictions, (len(predictions), 16, 1))

    predicted_3d_poses = np.concatenate((input_val_data, predictions), axis=2)
    predicted_3d_poses = np.concatenate((roots, predicted_3d_poses), axis=1)

    disparity_list = []
    error = []
    for i in range(len(predicted_3d_poses)):
        # Perform procrustes_alignment
        mtx1, mtx2, disparity = procrustes(ground_truth_val_data[i], predicted_3d_poses[i])
        error.append(disparity)
        #disparity_list.append(disparity)

    if np.mean(error) < best_error:
        best_error = np.mean(error)
        best_weight = weight_file
        best_predictions = predicted_3d_poses
    print('The average error for weights ' + weight_file + ' the validation set is %.3f MM' % np.mean(error))
    #print('The average disparity between gt and predictions is %.3f' % np.mean(disparity))
    print('The best error so far is ' + str(best_error) + ' using weights ' + best_weight)
    print()

print('The best error was ' + str(best_error) + ' using weights ' + best_weight)
with open('predictions/amazonlab_model/batch_size_8192_binaryGAN_predicted_3d_poses.json', 'w') as outfile:
    json.dump(best_predictions.tolist(), outfile)
