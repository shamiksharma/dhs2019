from tensorflow import keras
from camera import Camera
import poselib
import numpy as np
import glob
import cv2
from tqdm import tqdm
from pose_matching import DeepPoseMatcher, SimplePoseMatcher, get_video_as_frames
from kalman_pose import PoseCorrector
import threading
from queue import LifoQueue, Queue
from collections import deque
import utils
import time

fast_mode = 'tpu'
accurate_mode = 'tpu'

class SlowMethodThread(threading.Thread):
    def __init__(self, slowmethod, callback):
        threading.Thread.__init__(self)
        self.q = LifoQueue(1000)
        self.slowmethod = slowmethod
        self.callback = callback

    def run(self):
        while True:
            args = self.get()
            if args is not None:
                response = self.slowmethod(args)
                if response is not None:
                    self.callback(response)

    def update(self, args):
        self.q.put(args)

    def get(self):
        o = self.q.get()
        self.q.queue.clear()
        return o

class Display:
    def __init__(self, width):
        self.width = width

    def render(self, user_image, pose_image, score, elapsed):
        w,h,c = pose_image.shape
        user_image = cv2.resize(user_image, (h,w))
        combo = np.concatenate((user_image, pose_image), axis=1)
        w,h,c = combo.shape
        combo = cv2.resize(combo, (int(h*(self.width/w)), self.width))

        self.centered_text(combo, str(int(score)))
        self.centered_text(combo, str(int(elapsed)), 200, (128, 255, 128))

        cv2.imshow("display", combo)
        cv2.waitKey(10)


    def centered_text(self, image, text, y_pad = 0, color=(255, 0, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(text, font, 2, 2)[0]
        textX = (image.shape[1] - textsize[0]) // 2
        textY = (image.shape[0] + textsize[1]) // 2 + y_pad
        padding = 10
        cv2.rectangle(image, (textX-padding, textY+padding), (textX+textsize[0]+padding, textY - textsize[1]-padding),
                      thickness=-1, color=(0,0,0))
        cv2.putText(image, text, (textX, textY), font, 2, color, lineType=cv2.LINE_AA, thickness=2)


def get_target_poses(images, detector):
    poses = []
    good_images = []
    keypoints = []
    tscores = []
    for image in tqdm(images):
        image, flag, kp, scores = detector.detect(image, crop=False, pad=True)
        pose = utils.pose_scores_to_vector(kp, scores)
        # image = detector.draw(image, kp, scores)
        if pose is None:
            continue

        poses.append(pose)
        good_images.append(image)
        keypoints.append(kp)
        tscores.append(scores)

    return good_images, poses, keypoints, tscores

class Timer:
    def __init__(self):
        self.start = time.time()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

def exerciser(images, matcher_model, kalman_model, video):
    fast_detector = poselib.PoseDetector(fast_mode)
    accurate_detector = poselib.PoseDetector(accurate_mode)

    matcher = DeepPoseMatcher(poses=None, model_path=matcher_model)
    target_images, target_poses, target_keypoints, target_scores = get_target_poses(images, accurate_detector)

    if str.isdigit(video):
        video = int(video)

    cam = Camera(video, 10)
    cam.start()

    temporal_correction = None
    slow_temporal_correction = None

    if kalman_model:
        temporal_correction = PoseCorrector(kalman_model)
        slow_temporal_correction = SlowMethodThread(accurate_detector, temporal_correction.update)
        slow_temporal_correction.start()

    target_index = 0

    display = Display(600)
    score_deque_size = 1
    scores = deque([0 for i in range(score_deque_size)])
    n_frames_pose = 0
    max_frames_pose = 100

    timer = Timer()
    timer.reset()

    for i in tqdm(range(10000)):
        target_pose, target_image, target_keypoint, target_score = target_poses[target_index], \
                                                                   target_images[target_index],\
                                                                   target_keypoints[target_index],\
                                                                   target_scores[target_index]
        target_image = np.copy(target_image)
        image, count = cam.get()
        image = cv2.flip(image, 1)
        image, flag, keypoints, kp_scores = fast_detector.detect(image, crop=True, pad=False)
        fast_detector.draw(target_image, keypoints, kp_scores)
        fast_detector.draw(image, target_keypoint, target_score)

        pose = utils.pose_scores_to_vector(keypoints, kp_scores)

        if pose is None:
            score = 0
        else:
            if kalman_model:
                slow_temporal_correction.update(pose)
                corrected_pose = temporal_correction.update(pose)
                pose = corrected_pose
            score = matcher.similarity(pose, target_pose)

        scores.appendleft(score)
        scores.pop()

        average_score = sum(scores) / len(scores)
        display.render(image, target_image, average_score*100, timer.elapsed())
        n_frames_pose += 1
        if timer.elapsed() > 7:
            timer.reset()
            n_frames_pose = 0
            target_index += 1
            scores = deque([0 for i in range(score_deque_size)])
            continue

        # if average_score > .90:
        #     timer.reset()
        #     target_index += 1
        #     scores = deque([0 for i in range(score_deque_size)])

    cam.stop()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=None, required=True)
    parser.add_argument("--matcher", default=None, required=True)
    parser.add_argument("--kalman", default=None, required=False)
    parser.add_argument("--video", default=None, required=False)

    args = parser.parse_args()

    file_paths = list(glob.glob(args.images + "/*"))
    images = [cv2.imread(f) for f in sorted(file_paths)]
    if len(images) == 0:
        images = get_video_as_frames(args.images)[:1000]
    print("Num Images", len(images))

    exerciser(images, args.matcher, args.kalman, args.video)
