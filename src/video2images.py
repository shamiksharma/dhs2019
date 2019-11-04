import cv2
from tqdm import tqdm

def video2images(video, output):
    capture = cv2.VideoCapture(video)
    for i in tqdm(range(10000)):
        retval, image = capture.read()
        if not retval:
            break

        cv2.imwrite(, image)
        cv2.imshow("show", detector.draw(image, kp, scores))
        cv2.waitKey(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, required=True)
    parser.add_argument("--output", default=None, required=True)
    parser.add_argument("--mode", default=None, required=True)

    args = parser.parse_args()

    video2pose(args.video, args.output, args.mode)