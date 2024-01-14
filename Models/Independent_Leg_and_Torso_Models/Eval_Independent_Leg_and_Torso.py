import json
import numpy as np
import Label_Smoothing_LT_Model as Leg_And_Torso_GAN
import os
import pandas as pd
import tensorflow as tf
import keras.backend as K


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


"""
This file is used to perform procrustes alignment to the predicted output. 
What this does is scale and rotate the predicted 3D pose from the generator the minimise the distance 
between this pose and the ground truth 3D pose. No manipulation is done this is purely scaling and rotation as 
the predicted 3D pose is no to scale with the ground truth 3D pose and therefore a direct comparison is not 
feasible
"""

torso_input_size = (10, 2)
legs_input_size = (6, 2)# The input to the generator and discriminator [16 (x,y) keypoints]
loss_type = 'MSE'
model_number = '1'
generator_weight_files = '../../model_weights/leg_and_torso_weights/'

test_file = '../../EVAL_DATA/final_testing_data.json'

with open(test_file) as f:
    test_data = np.array(json.load(f))


ground_truth_val_data = np.array([t['3d_pose'][1:] for t in test_data])
action_list = [t['action'] for t in test_data]
input_val_data = tf.convert_to_tensor([t['norm_2d_pose'] for t in test_data])


# Build generator and load in its weights
legs_generator = Leg_And_Torso_GAN.build_legs_generator(legs_input_size)
torso_generator = Leg_And_Torso_GAN.build_torso_generator(torso_input_size)
all_weights_list = os.listdir(generator_weight_files)
legs_weights = []
torso_weights = []
for weight_file in all_weights_list:
    if 'lower' in weight_file:
        legs_weights.append(weight_file)
    else:
        torso_weights.append(weight_file)

legs_weights.sort()
torso_weights.sort()

all_errors = []
best_error = 99999

all_errors = []
epoch = 1
for i in range(len(legs_weights)):
    legs_generator.load_weights(generator_weight_files + legs_weights[i])
    torso_generator.load_weights(generator_weight_files + torso_weights[i])
    torso_predictions = torso_generator.predict(input_val_data[:, 6:, :], batch_size=8192)
    legs_predictions = legs_generator.predict(input_val_data[:, :6, :], batch_size=8192)
    predictions = np.concatenate((legs_predictions, torso_predictions), axis=1)
    predictions = np.reshape(predictions, (len(predictions), 16, 1))
    predicted_3d_poses = np.concatenate((input_val_data, predictions), axis=2)
    error_list = []
    for j in range(len(predicted_3d_poses)):
        # Perform procrustes_alignment
        d, Z, tform = procrustes(ground_truth_val_data[j], predicted_3d_poses[j])
        error = (np.mean(np.sqrt(np.sum((ground_truth_val_data[j] - Z) ** 2, axis=1))))
        error_list.append(error)
    all_errors.append(np.mean(error_list))

    if np.mean(error_list) < best_error:
        best_error = np.mean(error_list)
        best_epoch = i + 1
        best_predictions = predicted_3d_poses

    print('The average error for epoch ' + str(i + 1) + ' on the validation set is %.5f MM' % np.mean(error_list))
    print('The lowest error so far is ' + str(best_error) + ' using epoch ' + str(best_epoch))
    print()
    K.clear_session()


all_errors = pd.DataFrame(all_errors)
all_errors.to_csv('independent_leg_and_torso_errors_' + loss_type + '_' + model_number + '.csv', index=False)
