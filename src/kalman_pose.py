"""
pretrained model
"""

from tensorflow import keras
import  cv2
import poselib
import numpy as np
from tqdm import tqdm
from tensorflow import keras

cache_dir = "cache/"

def get_video_as_frames(video_path):
    capture = cv2.VideoCapture(video_path)
    while True:
        retval, frame = capture.read()
        if not retval:
            return
        yield frame

def get_model():
    L = keras.layers
    seq = keras.models.Sequential()
    seq.add(L.ConvLSTM2D(filters=5, kernel_size=(3, 3),
                       input_shape=(None, 17, 3, 1),
                       padding='same', return_sequences=True, activation='relu'))

    seq.add(L.Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation='relu',
                   padding='same', data_format='channels_last'))

    seq.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])

    return seq

def create_sequence(poses, context_length):
    pose_batches = []
    for i in range(len(poses) - context_length):
        slice = poses[i:i + context_length]
        pose_batches.append(slice)

    pose_batches = np.asarray(pose_batches)
    return pose_batches


def lr_schedule():
    def lrs(epoch):
        lr = 0.0002
        if epoch >= 1: lr = 0.0001
        if epoch >= 5: lr = 0.00005
        if epoch >= 10: lr = 0.00001
        return lr
    return keras.callbacks.LearningRateScheduler(lrs, verbose=True)

def checkpoint(path):
    cp = keras.callbacks.ModelCheckpoint(filepath=path,
                                         monitor='val_loss',
                                         save_best_only=True,
                                         verbose=1)
    return cp

def train(videos, output):
    accurate_detector_name = 'openpose'
    fast_detector_name = 'tpu'

    accurate_detector = poselib.PoseDetector(accurate_detector_name)
    fast_detector = poselib.PoseDetector(fast_detector_name)

    cache = get_cache()

    all_acc_poses = []
    all_fas_poses = []

    for video in videos:
        acc_name = video.split("/")[-1].split(".")[0] + accurate_detector_name
        fas_name = video.split("/")[-1].split(".")[0] + fast_detector_name
        if acc_name in cache and fas_name in cache:
            acc_poses = cache[acc_name]
            fas_poses = cache[fas_name]
        else:
            acc_poses = []
            fas_poses = []
            for frame in tqdm(get_video_as_frames(video)):
                pacc = get_pose(accurate_detector, frame)
                pfast = get_pose(fast_detector, frame)
                if pacc is None or pfast is None:
                    continue
                acc_poses.append(pacc)
                fas_poses.append(pfast)

            acc_poses = np.asarray(acc_poses)
            fas_poses = np.asarray(fas_poses)

            set_cache(acc_name, acc_poses)
            set_cache(fas_name, fas_poses)

        all_acc_poses.append(acc_poses)
        all_fas_poses.append(fas_poses)

    model = get_model()
    print (model.summary())
    accurate_poses = []
    fast_poses = []

    context_length = 5
    for acc, fas in zip(all_acc_poses, all_fas_poses):
        pa = create_sequence(acc, context_length)
        pf = create_sequence(fas, context_length)

        accurate_poses.extend(pa)
        fast_poses.extend(pf)

    fast_poses = np.expand_dims(np.asarray(fast_poses), axis=-1)
    accurate_poses = np.expand_dims(np.asarray(accurate_poses), axis=-1)

    cp = checkpoint("../weights/kalman_model.hdf5")
    model = keras.models.load_model("../weights/kalman_model.hdf5")
    callbacks = [lr_schedule(), cp]
    model.fit(fast_poses,
              accurate_poses,
              epochs=100,
              validation_split=0.2, callbacks=callbacks)

def get_cache():
    cache = {}
    for fname in glob.glob(cache_dir + "*.npy"):
        var_name = fname.split("/")[-1].split(".")[0]
        cache[var_name] = np.load(fname)
    return cache

def set_cache(var_name, arr):
    fname = cache_dir + var_name + ".npy"
    np.save(fname, arr)

class PoseCorrector:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    def update(self, pose):
        pose = np.expand_dims(pose, axis=-1)
        predicted = self.model.predict(pose)
        return predicted


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default=None, required=True)
    parser.add_argument("--output", default=None, required=True)
    args = parser.parse_args()
    import glob
    video_paths = glob.glob(args.videos + "*.*")
    print (video_paths)
    train(video_paths, args.output)