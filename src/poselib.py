import cv2
from tqdm import tqdm
from edgetpu.basic.basic_engine import BasicEngine
from tflitemodel import TFLiteModel
import numpy as np

pose_detection_tpu = "../weights/tpumodels/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite"
pose_detection_cpu = "../weights/tpumodels/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"

openpose_proto = "../weights/openpose/coco/pose_deploy_linevec.prototxt"
openpose_model = "../weights/openpose/coco/pose_iter_440000.caffemodel"

segmentation_cpu_10fps = "../weights/mnv2_10fps_unet/quant.tflite"
segmentation_cpu_30fps = "../weights/mnv2_unet_128/quant.tflite"


GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255,255,255)

KEYPOINTS = (
    'nose',
    'left eye',
    'right eye',
    'left ear',
    'right ear',
    'left shoulder',
    'right shoulder',
    'left elbow',
    'right elbow',
    'left wrist',
    'right wrist',
    'left hip',
    'right hip',
    'left knee',
    'right knee',
    'left ankle',
    'right ankle'
)

EDGES = (
    ('nose', 'left eye'),
    ('nose', 'right eye'),
    ('nose', 'left ear'),
    ('nose', 'right ear'),
    ('left ear', 'left eye'),
    ('right ear', 'right eye'),
    ('left eye', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right elbow'),
    ('right shoulder', 'right hip'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip'),
    ('left hip', 'left knee'),
    ('right hip', 'right knee'),
    ('left knee', 'left ankle'),
    ('right knee', 'right ankle'),
)

kp_indices = {k:i for i,k in enumerate(KEYPOINTS)}
edge_kp_pairs = []
for e1, e2 in EDGES:
    edge_kp_pairs.append((kp_indices[e1], kp_indices[e2]))


tf_coco_partmap = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
tf_openpose_partmap = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]


def crop(image, width, height):
    """
    Crops an image to desired width / height ratio
    :param image: image to crop
    :param width: desired width
    :param height: desired height
    :return: returns an image cropped to width/height ratio
    """
    desired_ratio = width / height
    image_width = image.shape[1]
    image_height = image.shape[0]
    image_ratio = image_width / image_height
    new_width, new_height = image_width, image_height

    # if original image is wider than desired image, crop across width
    if image_ratio > desired_ratio:
        new_width = int(image_height * desired_ratio)

    # crop across height otherwise
    elif image_ratio < desired_ratio:
        new_height = int(image_width / desired_ratio)

    image = image[image_height // 2 - new_height // 2: image_height // 2 + new_height // 2,
            image_width // 2 - new_width // 2: image_width // 2 + new_width // 2]

    return image

class OpenCVDetector:
    def __init__(self, proto, model, width, height, scale):
        self.net = cv2.dnn.readNet(cv2.samples.findFile(proto), cv2.samples.findFile(model))
        self.width = width
        self.height = height
        self.scale = scale

    def detect(self, image):
        frameWidth = image.shape[1]
        frameHeight = image.shape[0]
        inp = cv2.dnn.blobFromImage(image, self.scale, (self.width, self.height),
                                    (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(inp)
        out = self.net.forward()

        points = []
        scores = []
        for i in tf_coco_partmap:
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            points.append((int(x), int(y)))
            scores.append(conf)

        return True, points, scores

class TPUPose(BasicEngine):
    def __init__(self, model_path, mirror=False):
        BasicEngine.__init__(self, model_path)
        self._mirror = mirror

        self._input_tensor_shape = self.get_input_tensor_shape()
        _, self.image_height, self.image_width, self.image_depth = self.get_input_tensor_shape()

        offset = 0
        self._output_offsets = [0]
        for size in self.get_all_output_tensors_sizes():
            offset += size
            self._output_offsets.append(offset)

    def detect(self, img):
        assert (img.shape == tuple(self._input_tensor_shape[1:]))

        # Run the inference (API expects the data to be flattened)
        inference_time, output = self.run_inference(img.flatten())
        outputs = [output[i:j] for i, j in zip(self._output_offsets, self._output_offsets[1:])]

        keypoints = outputs[0].reshape(-1, len(KEYPOINTS), 2)
        keypoint_scores = outputs[1].reshape(-1, len(KEYPOINTS))
        nposes = int(outputs[3][0])
        assert nposes < outputs[0].shape[0]
        return keypoints[:nposes], keypoint_scores[:nposes]

class TPUPoseLib:
    def __init__(self):
        self.pose = TPUPose(pose_detection_tpu)
        self.height = self.pose.image_height
        self.width = self.pose.image_width
        print ("Initialized a model expecting (w,h)", self.width, self.height)

    def detect(self, original_image_int):
        pose_image_int = cv2.cvtColor(original_image_int, cv2.COLOR_BGR2RGB)
        pose_flag, pose1_kp, pose1_scores = False, None, None
        pose_image_int = crop(pose_image_int, self.pose.image_width, self.pose.image_height)
        pose_image_int = cv2.resize(pose_image_int, (self.width, self.height))
        poses_kp, poses_scores = self.pose.detect(pose_image_int)

        if poses_kp is not None and len(poses_kp) > 0:
            pose_flag = True
            pose1_kp, pose1_scores = poses_kp[0], poses_scores[0]
            pose1_kp = [(int(y), int(x)) for x, y in pose1_kp]
            pose1_kp = self.rescale_keypoints(pose1_kp, pose_image_int.shape)
            pose1_scores = list(map(float, pose1_scores))
        return pose_flag, pose1_kp, pose1_scores

    def rescale_keypoints(self, keypoints, shape):
        scale_x = shape[1] / self.width
        scale_y = shape[0] / self.height
        keypoints = [(int(i[0] * scale_x), int(i[1] * scale_y)) for i in keypoints]
        return keypoints

class OpenPoseDetector:
    def __init__(self):
        import sys
        sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
        self.op = op
        params = {"model_folder": "/home/apurva/work/libraries/openpose/models/"}
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

    def detect(self, image):
        flag, keypoints, scores = True, None, None
        if image is None:
            return flag, keypoints, scores

        datum = self.op.Datum()
        datum.cvInputData = image
        self.opWrapper.emplaceAndPop([datum])
        results = datum.poseKeypoints
        if results.shape:
            keypoints, scores = [], []
            for index in tf_openpose_partmap:
                r = results[0][index]
                keypoints.append((r[0], r[1]))
                scores.append(r[2])
        return True, keypoints, scores

class PoseDetector:
    def __init__(self, mode='tpu'):
        self.w = 257
        self.h = 257
        if mode == 'tpu':
            self.detector = TPUPoseLib()
        elif mode == 'cpu':
            self.detector = CPUTFLitePoseLib()
        else:
            self.detector = OpenPoseDetector()


    def prepare_image(self, frame, flip=True):
        if flip:
            frame = cv2.flip(frame, 1)

        frame = crop(frame, self.w, self.h)
        frame = cv2.resize(frame, (self.w, self.h))
        return frame

    def detect(self, image, crop=True):
        if crop:
            image = self.prepare_image(image)
        flag, kp, score = self.detector.detect(image)
        return image, flag, kp, score

    def draw(self, image, keypoints, scores):
        if keypoints is None:
            return image

        for point, score in zip(keypoints, scores):
            x, y = point
            color = GREEN
            if score < 0.5:
                color = RED
            cv2.circle(image, (x, y), 3, color, -1)

        for node_i, node_j in edge_kp_pairs:
            pt1 = keypoints[node_i]
            pt2 = keypoints[node_j]
            if scores[node_i] > 0.5 and scores[node_j] > 0.5:
                cv2.line(image, pt1, pt2, WHITE, 1, lineType=cv2.LINE_AA)


        return image

class PersonSegmentation(TFLiteModel):
    def __init__(self, fast, model_path=None):
        if model_path is None:
            if fast:
                model_path = segmentation_cpu_30fps
            else:
                model_path = segmentation_cpu_10fps

        self.load_model(model_path)

    def segment(self, image):
        imshape = image.shape
        image = self.preprocess(image)
        interpretation = self.get_model_output(image)
        heatmap = interpretation[0]
        heatmap = cv2.resize(heatmap, (imshape[1], imshape[0]))
        return heatmap

class CPUTFLitePoseLib(TFLiteModel):
    def __init__(self):
        model_path = pose_detection_cpu
        self.load_model(model_path)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def detect(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow("test", image)
        image = self.preprocess(image)
        interpretation = self.get_model_output(image)
        heatmap, offsets = interpretation[0], interpretation[1]
        # heatmap = self.sigmoid(heatmap)
        channels = heatmap.shape[-1]
        points = []
        scores = []
        size = heatmap.shape[0]
        r = float(self.width / (size - 1) )

        for i in range(channels):
            _, conf, _, point = cv2.minMaxLoc(heatmap[:,:,i])
            offset = (offsets[point[1], point[0]][i], offsets[point[1],point[0]][i + channels])
            keypoint = (int(point[0]*r + offset[1]), int(point[1]*r + offset[0]))
            points.append(keypoint)
            scores.append(self.sigmoid(conf))

        return True, points, scores

def demo(cam, mode='tpu'):
    detector = PoseDetector(mode)
    # segmenter = PersonSegmentation(False)
    cam.start()

    for i in tqdm(range(10000)):
        frame, count = cam.get()
        image, pose_flag, keypoints, scores = detector.detect(frame)
        # heatmap = segmenter.segment(image)
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR) / 255.
        # image = image * heatmap
        image = detector.draw(image, keypoints, scores)
        cv2.imshow("win1", image)
        cv2.waitKey(1)

if __name__ == "__main__":
    from camera import Camera
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, required=True)
    parser.add_argument("--mode", default=None, required=True)

    args = parser.parse_args()
    path = args.path

    if str.isdigit(args.path):
        path = int(args.path)

    cam = Camera(path, 200)
    demo(cam, args.mode)

