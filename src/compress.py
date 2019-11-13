import os, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

import tensorflow as tf
from tensorflow import lite
from utils import DataGenerator, get_train_data

def compress(saved_model_path, tflite_model_path, img_size, quantize=None, device=None):
    converter = lite.TFLiteConverter.from_saved_model(saved_model_path)

    if quantize:
        sample_dataset = DataGenerator(get_train_data(), 10, img_size).sample()
        sample_images = sample_dataset[0]

        def representative_dataset_gen():
          for index in range(sample_images.shape[0] - 1):
            yield [sample_images[index:index+1]]

        converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    x = open(tflite_model_path, "wb").write(tflite_model)
    print (x)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model", default=None, required=True)
    parser.add_argument("--output", default=None, required=True)
    parser.add_argument("--imgsize", default=128, required=False)


    args = parser.parse_args()
    compress(args.saved_model, args.output, int(args.imgsize), quantize=False)