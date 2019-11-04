from tensorflow_core.lite.python.interpreter import Interpreter
import cv2
import numpy as np

class TFLiteModel:
    def load_model(self, model_path):
        self.interpreter = Interpreter(
            model_path=model_path,)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        return self

    def preprocess(self, image):
        image = cv2.resize(image, (self.height, self.width))
        image = (np.asarray(image, dtype=np.float32) - 127.5) / 127.5
        image = np.expand_dims(image, axis=0)
        return image

    def get_model_output(self, image):
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        outputs = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]
        outputs = [np.squeeze(x) for x in outputs]
        return outputs

    def get_model_details(self):
        return self.interpreter.get_tensor_details()

    def model_path(self):
        return None

if __name__ == "__main__":
    import argparse, os, glob, traceback
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="pass either a directory or a tflite model", required=True)
    parser.add_argument("--image", default=None, help="pass a sample image", required=False)

    args = parser.parse_args()

    image = None
    if args.image:
        image = cv2.imread(args.image)

    models = []
    if os.path.exists(args.model):
        if os.path.isfile(args.model):
            models.append(args.model)
        elif os.path.isdir(args.model):
            files = glob.glob(args.model + "/*.tflite")
            models.extend(files)

    print (models)
    failed_models = []
    for path in models:
        try:
            tfutil = TFLiteModel().load_model(path)
            print (f"--------{path}---------")
            print ([(i['name'], i['shape']) for i in tfutil.input_details])
            print ([(i['name'], i['shape']) for i in tfutil.output_details])
            for x in tfutil.get_model_details():
                print (x)

            if image is not None:
                output = tfutil.get_model_output(image)
                print (output)
        except:
            failed_models.append(path)
            print("Exception in user code:")
            print("-" * 60)
            traceback.print_exc()
            print("-" * 60)

    print("----------Failed------------", failed_models)
