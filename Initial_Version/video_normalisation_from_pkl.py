import pickle
import json
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# amount of frames in each sequence for that particular dataset
max_sequence_len = 12

# preprocess training data
with open('h36m_train.pkl', 'rb') as f:
    train = pickle.load(f)

train_anno = [{} for sub in range(len(train))]

train_videos_num = 0
for i in range(len(train)):
    if train[i]['video_id'] > train_videos_num:
        train_videos_num = train[i]['video_id']
    train_anno[i]['video_id'] = train[i]['video_id']
    train_anno[i]['joints_3d'] = train[i]['joints_3d']

train = None

train_video_data = [[] for _ in range(train_videos_num + 1)]

for i in train_anno:
    video = i['video_id']
    joints = i['joints_3d']
    root = joints[0]
    joints = joints - root
    joints = joints / np.max(abs(joints))
    train_video_data[video].append(np.array(joints[1:, :2]).flatten())

train_data_sequences = []
for video in train_video_data:
    for frame in range(len(video)):
        sequence = []
        if frame < max_sequence_len:
            sequence.append(video[:frame + 1])
            train_data_sequences.append(pad_sequences(sequence, maxlen=max_sequence_len, padding='pre', dtype='float32'))
        else:
            sequence.append(np.array(video[(frame - max_sequence_len + 1):frame + 1]).reshape(12 ,32))
            train_data_sequences.append(np.asarray(sequence))

# pre-process validation data
with open('h36m_validation.pkl', 'rb') as f:
    test = pickle.load(f)

test_anno = [{} for sub in range(len(test))]

test_videos_num = 0
for i in range(len(test)):
    if test[i]['video_id'] > test_videos_num:
        test_videos_num = test[i]['video_id']
    test_anno[i]['video_id'] = test[i]['video_id']
    test_anno[i]['joints_3d'] = test[i]['joints_3d']

test = None

gt_test_video_data = [[] for _ in range(test_videos_num - train_videos_num)]
norm_test_video_data = [[] for _ in range(test_videos_num - train_videos_num)]
for i in test_anno:
    video = i['video_id'] - (train_videos_num + 1)
    joints = i['joints_3d']
    root = joints[0]
    joints = joints - root
    gt_test_video_data[video].append(np.array(joints))
    joints = joints / np.max(abs(joints))
    norm_test_video_data[video].append(np.array(joints[1:, :2] .flatten()))

gen_test_data_sequences = []
gt_test_data_sequences = []
for video in range(len(norm_test_video_data)):
    for frame in range(len(norm_test_video_data[video])):
        sequence = []
        if frame < max_sequence_len:
            sequence.append(norm_test_video_data[video][:frame + 1])
            gen_test_data_sequences.append(pad_sequences(sequence, maxlen=max_sequence_len, padding='pre', dtype='float32'))
        else:
            sequence.append(np.array(norm_test_video_data[video][(frame - max_sequence_len + 1):frame + 1]).reshape(12, 32))
            gen_test_data_sequences.append(np.asarray(sequence))
        gt_test_data_sequences.append(np.asarray(gt_test_video_data[video][frame]).reshape((17, 3)))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


with open('Datasets_For_TCN/' + str(max_sequence_len) + '_frames_h36m_train_video_anno.json', 'w') as outfile:
    json.dump(train_data_sequences, outfile, cls=NumpyEncoder)

with open('Datasets_For_TCN/' + str(max_sequence_len) + '_frames_h36m_test_video_anno.json', 'w') as outfile:
    json.dump(gen_test_data_sequences, outfile, cls=NumpyEncoder)

with open('Datasets_For_TCN/' + str(max_sequence_len) + '_frames_h36m_3dgt_test_video_anno.json', 'w') as outfile:
    json.dump(gt_test_data_sequences, outfile, cls=NumpyEncoder)

print('IM DONE!')
