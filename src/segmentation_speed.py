from tensorflow import keras
from poselib import PersonSegmentation
from camera import Camera
from tqdm import tqdm
from common import get_segmentation_model
import numpy as np
import cv2
import time

def runloop(image_size, video_path, big_model, small_model, slow=False):
    cam = Camera(video_path, 60)
    cam.start()
    start = time.time()
    for i in tqdm(range(200), desc="Running model ... "):
        frame, count = cam.get()
        frame_shape = frame.shape
        if slow:
            frame = cv2.resize(frame, (image_size, image_size)) / 255.
            frame = np.expand_dims(frame, axis=0)
            image = big_model.predict(frame)
            image = np.squeeze(image)
            image = cv2.resize(image, (frame_shape[1], frame_shape[0]))
        else:
            image = small_model.segment(frame)

        end = time.time()
        fps = int(i/(end - start))
        cv2.putText(image, str(fps) + "fps", (50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255,255,255), lineType=-1)
        cv2.imshow("model", image)
        cv2.waitKey(10)

    cam.stop()

def compare_speed(keras_model, tflite_model, video_path):
    image_size = 128
    big_model = get_segmentation_model(128)
    big_model.load_weights(keras_model)
    small_model = PersonSegmentation(True, tflite_model)
    runloop(image_size, video_path, big_model, small_model, slow=False)
    cv2.waitKey(-1)
    runloop(image_size, video_path, big_model, small_model, slow=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--keras", default=None, required=True)
    parser.add_argument("--tflite", default=None, required=True)
    parser.add_argument("--video", default=None, required=True)

    args = parser.parse_args()

    compare_speed(args.keras, args.tflite, args.video)