import os, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

import segmentation_models as sm
sm.set_framework('tf.keras')

from utils import (data_path, Display, get_train_data, DisplayCallback, DataGenerator)
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import tensorflow as tf

def lr_schedule():
    def lrs(epoch):
        lr = 0.0002
        if epoch >= 1: lr = 0.0001
        if epoch >= 5: lr = 0.00005
        if epoch >= 10: lr = 0.00001
        return lr
    return keras.callbacks.LearningRateScheduler(lrs, verbose=True)

def train(output_dir):
    img_size = 128
    batch_size = 16

    model = sm.Unet('mobilenetv2',
                    input_shape=(img_size, img_size, 3),
                    classes=1,
                    activation='sigmoid',
                    decoder_filters=(128, 64, 32, 32, 16),
                    encoder_weights='imagenet')

    print (model.summary())

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=sm.losses.bce_jaccard_loss,
                  metrics=[sm.metrics.iou_score])


    data = get_train_data()
    val_split = 0.1
    train_size = int(len(data)*(1 - val_split))
    train_data = data[:train_size]
    val_data = data[train_size:]

    train_generator = DataGenerator(train_data, batch_size, img_size)
    val_generator = DataGenerator(val_data, batch_size, img_size)

    sample_batch = val_generator.sample()
    display_callback = DisplayCallback(model, sample_batch, img_size, 600, 100, frequency=50)

    model_path = output_dir + "/weights.hdf5"
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                       monitor='val_loss',
                                       save_best_only=True,
                                       verbose=1)

    callbacks = [lr_schedule(), cp, display_callback]

    history = model.fit_generator(train_generator,
                        validation_data = val_generator,
                        epochs=5,
                        shuffle=True,
                        callbacks=callbacks,
                        verbose=1,
                        use_multiprocessing=True,
                        workers=4)

    model.load_weights(model_path)
    tf.saved_model.save(model, output_dir + "/saved_model/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=None, required=True)

    args = parser.parse_args()
    train(args.outdir)