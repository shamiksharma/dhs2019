from tensorflow import keras
from camera import Camera
import poselib
import numpy as np
import glob
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import utils
from sklearn.preprocessing import normalize

class DeepPoseMatcher:
    def __init__(self, poses, model_path):
        tmp_model = keras.models.load_model(model_path)
        self.model = extract_embedder(tmp_model)

        if poses:
            poses = np.asarray(poses)
            self.embeddings = self.model.predict(poses)

    def match(self, pose):
        pose = np.expand_dims(pose, axis=0)
        embedding = self.model.predict(pose)
        distances = cosine_similarity(embedding, self.embeddings)
        best_index = np.argmax(distances)
        best_score = np.max(distances)
        return int(best_index), best_score

    def similarity(self, pose1, pose2):
        pose1 = np.expand_dims(pose1, axis=0)
        pose2 = np.expand_dims(pose2, axis=0)
        embedding1 = self.model.predict(pose1)
        embedding2 = self.model.predict(pose2)
        return cosine_similarity(embedding1, embedding2)[0][0]

class SimplePoseMatcher:
    def __init__(self, poses):
        if poses:
            self.poses = poses

    def get_pose_score(self, pose_scores):
        poses = np.asarray(pose_scores)[:,:,0:2]
        poses = np.reshape(poses, (poses.shape[0], poses.shape[1]*poses.shape[2], ))
        scores = np.asarray(pose_scores)[:, :, 2]
        return poses, scores

    def match(self, pose):
        scores = [self.similarity(pose, tpose) for tpose in self.poses]
        best_index = np.argmax(scores)
        best_score = np.max(scores)
        print(best_score, best_index)
        return int(best_index), best_score

    def similarity(self, pose1, pose2):
        pose1, score1 = self.get_pose_score(np.expand_dims(pose1, axis=0))
        pose2, score2 = self.get_pose_score(np.expand_dims(pose2, axis=0))
        conf_mult = (sum(score1[0]) + sum(score2[0]))/(len(score1[0]) + len(score2[0]))
        print (conf_mult)
        distances = cosine_similarity(pose1, pose2)
        return distances


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

def test_matching(images, mode, model, video):
    detector = poselib.PoseDetector(mode)
    poses = []
    good_images = []

    for image in tqdm(images):
        image, flag, kp, scores = detector.detect(image, crop=False, pad=True)
        pose = utils.pose_scores_to_vector(kp, scores)
        image = detector.draw(image, kp, scores)

        if pose is None:
            continue

        poses.append(pose)
        good_images.append(image)

    images = good_images

    matcher = DeepPoseMatcher(poses, model_path=model)

    if str.isdigit(video):
        video = int(video)

    cam = Camera(video, 30)
    cam.start()

    for i in range(10000000):
        image, count = cam.get()
        # image = cv2.flip(image, 1)
        image, flag, kp, scores = detector.detect(image, crop=True, pad=False)
        pose = utils.pose_scores_to_vector(kp, scores)
        image = detector.draw(image, kp, scores)

        cv2.imshow("you", image)

        if pose is None:
            continue

        best_index, best_score = matcher.match(pose)

        cv2.imshow("candidate", images[best_index])


        cv2.waitKey(10)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=None, required=True)
    parser.add_argument("--mode", default=None, required=True)
    parser.add_argument("--model", default=None, required=True)
    parser.add_argument("--video", default=None, required=True)

    args = parser.parse_args()

    file_paths = list(glob.glob(args.images + "/*"))
    images = [cv2.imread(f) for  f in file_paths]
    if len(images) == 0:
        images = get_video_as_frames(args.images)[:1000]
    print ("Num Images", len(images))
    test_matching(images, args.mode, args.model, args.video)