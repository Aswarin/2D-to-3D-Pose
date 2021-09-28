import pickle
import json
from collections import defaultdict
import numpy as np

with open('/home/aswarin/Desktop/2D To 3D Pose Lifting/datasets/h36m_train.pkl', 'rb') as f:
    train = pickle.load(f)

train_anno = [{} for sub in range(len(train))]

for i in range(len(train)):
    train_anno[i]['video_id'] = train[i]['video_id']
    train_anno[i]['joints_3d'] = train[i]['joints_3d']

train = None

with open('/home/aswarin/Desktop/2D To 3D Pose Lifting/datasets/h36m_validation.pkl', 'rb') as f:
    test = pickle.load(f)

test_anno = [{} for sub in range(len(test))]

for i in range(len(test)):
    test_anno[i]['video_id'] = test[i]['video_id']
    test_anno[i]['joints_3d'] = test[i]['joints_3d']

test = None

train_video_data = defaultdict(list)

for i in train_anno:
    key = i['video_id']
    joints = i['joints_3d']
    root = joints[0]
    joints = joints - root
    train_video_data[key].append(np.array(joints[1:]).flatten())

test_video_data = defaultdict(list)
for i in test_anno:
    key = i['video_id']
    joints = i['joints_3d']
    root = joints[0]
    joints = joints - root
    test_video_data[key].append(np.array(joints).flatten())


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


with open('h36m_train_video_anno.json', 'w') as outfile:
    json.dump(train_video_data, outfile, cls=NumpyEncoder)

with open('h36m_test_video_anno.json', 'w') as outfile:
    json.dump(test_video_data, outfile, cls=NumpyEncoder)
