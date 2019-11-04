import cv2
from tensorflow import keras
import poselib
import json
from tqdm import tqdm

def video2pose(video, output, mode):
    detector = poselib.PoseDetector(mode)
    poses = []
    capture = cv2.VideoCapture(video)
    for i in tqdm(range(10000)):
        retval, image = capture.read()
        if not retval:
            break

        image, flag, kp, scores = detector.detect(image)
        cv2.imshow("show", detector.draw(image, kp, scores))
        cv2.waitKey(1)
        if kp is not None:
            poses.append({"keypoints":kp, "scores":scores})

    json.dump(poses, open(output, 'w'), indent=2)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, required=True)
    parser.add_argument("--output", default=None, required=True)
    parser.add_argument("--mode", default=None, required=True)

    args = parser.parse_args()

    video2pose(args.video, args.output, args.mode)