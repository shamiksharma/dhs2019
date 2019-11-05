import cv2
from tensorflow import keras
import poselib
from tqdm import tqdm
import glob
import  numpy as np
import random
from sklearn import  utils
import tensorflow as tf

cache_dir = "cache/"

def get_pose(detector, image):
    image, flag, kp, scores = detector.detect(image, crop=False)
    if kp is None:
        return None
    kp = np.asarray(kp)
    kp[:, 0] = (kp[:, 0] - min(kp[:, 0]))/ (max(kp[:, 0]) - min(kp[:, 0]))
    kp[:, 1] = (kp[:, 1] - min(kp[:, 1]))/ (max(kp[:, 1]) - min(kp[:, 1]))
    scores = np.expand_dims(np.asarray(scores), axis=-1)
    kp = np.hstack((kp, scores))
    return kp

def prepare_data(images_dir, mode='openpose', use_cache=True, num_samples=None):
    x_name, y_name= "x", "y"

    if use_cache:
        cache = get_cache()
        if x_name in cache and y_name in cache:
            X = cache[x_name]
            y = cache[y_name]
            return X, y

    X, y = [], []
    file_paths = list(glob.glob(images_dir + "/*/*"))
    random.shuffle(file_paths)

    if num_samples:
        file_paths = file_paths[:num_samples]

    detector = poselib.PoseDetector(mode)

    for image_path in tqdm(file_paths):
        class_name = image_path.split("/")[-2]
        image = cv2.imread(image_path)
        if image is None:
            continue
        kp = get_pose(detector, image)
        if kp is None:
            continue
        y.append(class_name)
        X.append(kp)

    class_names_to_indices = {name:index for index, name in enumerate(set(y))}
    y = [class_names_to_indices[name] for name in y]
    X = np.asarray(X)
    y = np.asarray(y)
    print (X)
    cache = {x_name:X, y_name:y}
    set_cache(cache)
    return X,y


def get_model(n_classes, input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(128, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(n_classes, activation='softmax', use_bias=False))
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def get_cache():
    cache = {}
    for fname in glob.glob(cache_dir + "*.npy"):
        var_name = fname.split("/")[-1].split(".")[0]
        cache[var_name] = np.load(fname)
    return cache

def set_cache(cache):
    for var_name in cache:
        fname = cache_dir + var_name + ".npy"
        np.save(fname, cache[var_name])

def shuffle(arrays):
    size = arrays[0].shape[0]
    permutation = list(range(size))
    random.shuffle(permutation)
    arrays = [arr[permutation] for arr in arrays]
    return arrays

def lr_schedule():
    def lrs(epoch):
        lr = 0.0002
        if epoch >= 15: lr = 0.0001
        if epoch >= 45: lr = 0.00005
        if epoch >= 60: lr = 0.00001
        return lr
    return keras.callbacks.LearningRateScheduler(lrs, verbose=True)


def checkpoint(path):
    cp = keras.callbacks.ModelCheckpoint(filepath=path,
                                            monitor='val_loss',
                                            save_best_only=True,
                                            verbose=1)
    return cp

def test(images_dir, mode, output):
    X, y  = prepare_data(images_dir, mode, use_cache=False)
    n_classes = len(np.unique(y))
    num_features = X.shape[1:]
    class_weights = utils.class_weight.compute_class_weight('balanced',
                                                      np.unique(y),
                                                      y)

    model = get_model(n_classes, num_features)
    print (model.summary())
    callbacks = [lr_schedule(), checkpoint(output + "pose_metric.hdf5")]
    model.fit(X, y,
              validation_split=0.2,
              epochs=100,
              batch_size=4,
              class_weight=class_weights, callbacks=callbacks)

    keras.models.save_model(model, output + "pose_metric_model.hdf5")
    results = model.evaluate(X, y, verbose=0)
    print (results)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=None, required=True)
    parser.add_argument("--mode", default=None, required=True)
    parser.add_argument("--output", default=None, required=True)
    args = parser.parse_args()
    test(args.images, args.mode, args.output)