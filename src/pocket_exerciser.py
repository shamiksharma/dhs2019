from tensorflow import keras
from pose_classifier import get_pose
from camera import Camera
import poselib
import numpy as np
import glob
import cv2
from tqdm import tqdm
from pose_matching import DeepPoseMatcher
from kalman_pose import PoseCorrector
import threading
from queue import LifoQueue

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

    def render(self, user_image, pose_image, score):
        w,h,c = pose_image.shape
        user_image = cv2.resize(user_image, (h,w))
        combo = np.concatenate((user_image, pose_image), axis=1)
        w,h,c = combo.shape
        combo = cv2.resize(combo, (int(h*(self.width/w)), self.width))
        self.centered_text(combo, str(int(score*100)))
        cv2.imshow("display", combo)
        cv2.waitKey(10)


    def centered_text(self, image, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(text, font, 2, 2)[0]
        textX = (image.shape[1] - textsize[0]) // 2
        textY = (image.shape[0] + textsize[1]) // 2
        padding = 10
        cv2.rectangle(image, (textX-padding, textY+padding), (textX+textsize[0]+padding, textY - textsize[1]-padding),
                      thickness=-1, color=(0,0,0))
        cv2.putText(image, text, (textX, textY), font, 2, (255, 0, 255), lineType=cv2.LINE_AA, thickness=2)



def get_target_poses(images, detector):
    poses = []
    good_images = []

    for image in tqdm(images):
        pose = get_pose(detector, image)
        image = detector.prepare_image(image)
        if pose is None:
            continue

        poses.append(pose)
        good_images.append(image)

    return good_images, poses

def exerciser(images, matcher_model, kalman_model):
    fast_detector = poselib.PoseDetector(fast_mode)
    accurate_detector = poselib.PoseDetector(accurate_mode)

    matcher = DeepPoseMatcher(poses=None, model_path=matcher_model)
    target_images, target_poses = get_target_poses(images, accurate_detector)

    segmenter = poselib.PersonSegmentation(False)

    cam = Camera(0, 30)
    cam.start()

    temporal_correction = None
    slow_temporal_correction = None

    if kalman_model:
        temporal_correction = PoseCorrector(kalman_model)
        slow_temporal_correction = SlowMethodThread(accurate_detector, temporal_correction.update)
        slow_temporal_correction.start()

    target_index = 0

    display = Display(400)

    for i in range(10000000):
        target_pose, target_image = target_poses[target_index], target_images[target_index]
        image, count = cam.get()
        pose = get_pose(fast_detector, image)
        user_image = fast_detector.prepare_image(image)

        if pose is None:
            score = 0
        else:
            if kalman_model:
                slow_temporal_correction.update(pose)
                corrected_pose = temporal_correction.update(pose)
                pose = corrected_pose
            score = matcher.similarity(pose, target_pose)

        heatmap = segmenter.segment(user_image)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR) / 255.
        user_image = np.asarray(user_image * heatmap, dtype=np.uint8)

        display.render(user_image, target_image, score)
        if score > .96:
            target_index += 1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=None, required=True)
    parser.add_argument("--matcher", default=None, required=True)
    parser.add_argument("--kalman", default=None, required=False)
    args = parser.parse_args()

    file_paths = list(glob.glob(args.images + "/*"))
    images = [cv2.imread(f) for  f in file_paths]
    print ("Num Images", len(images))
    exerciser(images, args.matcher, args.kalman)