from tensorflow import keras
from yoga_classification import get_pose
from camera import Camera
import poselib
import numpy as np
import glob
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class DeepPoseMatcher:
    def __init__(self, poses, model_path):
        tmp_model = keras.models.load_model(model_path)
        self.model = extract_embedder(tmp_model)
        poses = np.asarray(poses)
        self.embeddings = self.model.predict(poses)

    def match(self, pose):
        pose = np.expand_dims(pose, axis=0)
        embedding = self.model.predict(pose)
        distances = cosine_similarity(embedding, self.embeddings)
        best_index = np.argmax(distances)
        return int(best_index)

class SimplePoseMatcher:
    def __init__(self, poses):
        self.poses, self.scores = self.get_pose_score(poses)

    def get_pose_score(self, pose_scores):
        poses = np.asarray(pose_scores)[:,:,0:2]
        poses = np.reshape(poses, (poses.shape[0], poses.shape[1]*poses.shape[2], ))
        scores = np.asarray(pose_scores)[:, :, 2]
        return poses, scores

    def match(self, pose):
        pose, score = self.get_pose_score(np.expand_dims(pose, axis=0))
        distances = cosine_similarity(pose, self.poses)
        best_index = np.argmax(distances)
        return int(best_index)

def extract_embedder(model):
    model = keras.models.Model(model.layers[0].input, model.layers[-2].output)
    model.summary()
    return model

def get_video_as_frames(video_path):
    capture = cv2.VideoCapture(video_path)
    frames = []
    while True:
        retval, frame = capture.read()
        if not retval:
            break
        frames.append(frame)

    return frames

def test_matching(images, mode, model):
    detector = poselib.PoseDetector(mode)
    poses = []
    good_images = []

    for image in tqdm(images):
        pose = get_pose(detector, image)
        if pose is None:
            continue
        poses.append(pose)
        good_images.append(image)

    images = good_images

    matcher = DeepPoseMatcher(poses, model)
    detector = poselib.PoseDetector(mode)
    cam = Camera(0, 30)
    cam.start()
    for i in range(10000):
        image, count = cam.get()
        pose = get_pose(detector, image)
        if pose is None:
            continue

        best_index = matcher.match(pose)
        cv2.imshow("candidate", images[best_index])
        cv2.imshow("you", image)

        cv2.waitKey(10)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=None, required=True)
    parser.add_argument("--mode", default=None, required=True)
    parser.add_argument("--model", default=None, required=True)
    args = parser.parse_args()

    file_paths = list(glob.glob(args.images + "/*/*"))
    images = [cv2.imread(f) for  f in file_paths]
    images = get_video_as_frames(args.images)
    print ("Num Images", len(images))
    test_matching(images, args.mode, args.model)