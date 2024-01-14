import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout, Add , Flatten, ReLU, BatchNormalization, Dense
from keras.models import Input, Model
import Models.Independent_Leg_and_Torso_Models.Label_Smoothing_LT_Model as Teacher
import numpy as np
import keras.metrics as M
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import pickle

# Residual Block from the paper (the paper uses relu and no dropout)
def residual_block(X, d_rate=0.5):
    X_shortcut = X

    X = Dense(1024)(X)
    X = BatchNormalization(momentum=0.9)(X)
    X = ReLU()(X)
    X = Dropout(d_rate)(X)

    X = Dense(1024)(X)
    X = BatchNormalization(momentum=0.9)(X)
    X = ReLU()(X)
    X = Dropout(d_rate)(X)

    X = Add()([X, X_shortcut])

    return X

def build_student_generator(input_shape):
    X_input = Input(input_shape)
    X = Flatten()(X_input)
    X = Dense(1024)(X)
    X = BatchNormalization(momentum=0.9)(X)
    X = ReLU()(X)
    X = residual_block(X, 0.5)
    X = residual_block(X, 0.5)
    X = residual_block(X, 0.5)
    X = Dense(16, activation='tanh')(X)  # 16 outputs (z coordinate for every keypoint)
    model = Model(inputs=X_input, outputs=X, name='Student')
    return model


class Callback(keras.callbacks.Callback):
    def __init__(self, gen_filepath=None, disc_filepath=None, GAN_filepath=None):
        self.gen_filepath = gen_filepath
        self.disc_filepath = disc_filepath
        self.GAN_filepath = GAN_filepath

    def on_epoch_end(self, epoch, logs=None):
        # Variable to control how many epochs to save weights
        N = 1
        if epoch >= 0:
            if self.gen_filepath != None:
                self.model.student.save_weights(self.gen_filepath + '%04d' % (epoch + 1) + '_student_weights.h5')
            if self.disc_filepath != None:
                self.model.discriminator.save_weights(self.disc_filepath + 'disc_weights%08d.h5' % (epoch + 1))
            if self.GAN_filepath != None:
                self.model.save_weights(self.GAN_filepath + 'GAN_weights%08d.h5' % (epoch + 1))


class Student_Model(Model):
    def __init__(self, student, torso_teacher, legs_teacher):
        super(Student_Model, self).__init__()
        self.student = student
        self.torso_teacher = torso_teacher
        self.legs_teacher = legs_teacher

    def compile(self, student_opt,
                mse):
        super(Student_Model, self).compile()
        self.student_opt = student_opt
        self.mse = mse
        self.student_error_metric = M.Mean(name='Student Error')


    def train_step(self, pose_batch):
        # Randomly flip the input pose batch along the y-axis
        if np.random.choice([0, 1]) == 0:
            x_pose = pose_batch[:, :, 0]
            y_pose = pose_batch[:, :, 1]
            x_pose *= -1
            pose_batch = tf.stack([x_pose, y_pose], axis=2)

        teacher_torso_predictions = self.torso_teacher(pose_batch[:, 6:, :])
        teacher_legs_predictions = self.legs_teacher(pose_batch[:, :6, :])
        teacher_predicted_z = tf.concat([teacher_legs_predictions, teacher_torso_predictions], axis=1)
        with tf.GradientTape() as tape:
            student_predicted_z = self.student(pose_batch)
            error = self.mse(student_predicted_z, teacher_predicted_z)
        gradients = tape.gradient(error, self.student.trainable_weights)
        self.student_opt.apply_gradients(zip(gradients, self.student.trainable_weights))
        self.student_error_metric.update_state(error)

        return {
            'loss':self.student_error_metric.result()
        }


if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)
    """Begin Actual Model---------------------------------------------------------------------------"""

    # Easy Parameter Tweaking
    batch_size = 8192
    student_input_size = (16, 2)
    disc_input_size_3d = (16, 3)
    disc_input_size_2d = (16, 2)
    leg_teacher_input_size = (6, 2)
    torso_teacher_input_size = (10, 2)  # The input to the generator and discriminator [16 (x,y) keypoints]'
    student_opt = Adam(learning_rate=0.0002)
    #student_opt = SGD(learning_rate=0.1, momentum=0.9)
    #student_opt = RMSprop()
    mean_squared_error = tf.keras.losses.MeanSquaredError()
    epochs = 500

    with open('../Normalisation and Data/h36m_train.pkl', 'rb') as f:
        train = pickle.load(f)

    train_data = []
    for t in train:
        keypoints = t['joints_3d']
        keypoints = keypoints - keypoints[0]
        two_d_keypoints = keypoints[:, :2]
        pose_max = np.max(abs(two_d_keypoints))
        normalised_two_d_keypoints = two_d_keypoints / pose_max
        train_data.append(normalised_two_d_keypoints[1:, :].astype('float32'))

    train = None

    # with open('correct_normalised_2d_train_poses.json') as f:
    #     train_data = json.load(f)
    #
    # with open('correct_normalised_2d_test_poses.json') as f:
    #     test_data = json.load(f)

    train_data = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)

    student = build_student_generator(student_input_size)
    torso_teacher = Teacher.build_upper_generator(torso_teacher_input_size)
    leg_teacher = Teacher.build_lower_generator(leg_teacher_input_size)
    torso_teacher.load_weights('Teacher/Teacher_Weights/upper_gen_weights.h5')
    leg_teacher.load_weights('Teacher/Teacher_Weights/lower_gen_weights.h5')
    torso_teacher.trainable = False
    leg_teacher.trainable = False
    gen_filepath = 'student_weights/'

    model = Student_Model(student=student, torso_teacher=torso_teacher, legs_teacher=leg_teacher)
    model.compile(student_opt=student_opt, mse=mean_squared_error)
    model.fit(train_data, epochs=epochs, verbose=2, callbacks=[Callback(gen_filepath=gen_filepath)])

