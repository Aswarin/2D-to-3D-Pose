import os
import numpy as np
import json
from MSE_Split_Combine_Model import build_body_generator
import pandas as pd
import tensorflow as tf
import pickle

"""
This file is used to perform procrustes alignment to the predicted output. 
What this does is scale and rotate the predicted 3D pose from the generator the minimise the distance 
between this pose and the ground truth 3D pose. No manipulation is done this is purely scaling and rotation as 
the predicted 3D pose is no to scale with the ground truth 3D pose and therefore a direct comparison is not 
feasible
"""


def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform


input_size = (16, 2)  # The input to the generator and discriminator [16 (x,y) keypoints]
leg_torso_split_combine_model = build_body_generator(input_size)
loss_type = 'MSE'
model_number = '1'
leg_torso_split_combine_model_weight_location = 'split_combine_leg_torso_weights/' + loss_type + '_' + model_number + '/' # Load in the generator weights from a h5 file

with open('h36m_train_eval.pkl', 'rb') as f:
    train = pickle.load(f)


input_val_data = []
ground_truth_val_data = []
# Normalise the data and increase the amount of training data by 2 by flipping along the x axis
for t in train:
    keypoints = t['joints_3d']
    keypoints = keypoints - keypoints[0]
    keypoints = keypoints[1:]
    ground_truth_val_data.append(keypoints)
    two_d_keypoints = keypoints[:, :2]
    pose_max = np.max(abs(two_d_keypoints))
    normalised_two_d_keypoints = two_d_keypoints / pose_max
    input_val_data.append(normalised_two_d_keypoints.astype('float32'))

input_val_data = np.array(input_val_data)
ground_truth_val_data = np.array(ground_truth_val_data)

best_error = 99999
best_weight = None
# Build generator and load in its weights
weight_list = []
for weight_file in os.listdir(leg_torso_split_combine_model_weight_location):
    weight_list.append(weight_file)

weight_list.sort()
all_errors = []
for weight_file in weight_list:
    leg_torso_split_combine_model.load_weights(leg_torso_split_combine_model_weight_location + weight_file)
    predictions = leg_torso_split_combine_model.predict(input_val_data, batch_size=8192)
    predictions = np.reshape(predictions, (len(predictions), 16, 1))
    predicted_3d_poses = np.concatenate((input_val_data, predictions), axis=2)
    error = []
    for j in range(len(predicted_3d_poses)):
        # Perform procrustes_alignment
        d, Z, tform = procrustes(ground_truth_val_data[j], predicted_3d_poses[j])
        error.append(np.mean(np.sqrt(np.sum((ground_truth_val_data[j] - Z) ** 2, axis=1))))

    if np.mean(error) < best_error:
        best_error = np.mean(error)
        best_weight = weight_file
        best_predictions = predicted_3d_poses
    all_errors.append(np.mean(error))

    print('The average error for weights ' + weight_file + ' the validation set is %.3f MM' % np.mean(error))
    print('The best error so far is ' + str(best_error) + ' using weights ' + str(best_weight))
    print()

all_errors = pd.DataFrame(all_errors)
all_errors.to_csv('leg_torso_split_combine_model_errors' + loss_type + '_' + model_number + '.csv', index=False)
