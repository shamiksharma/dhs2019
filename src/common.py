import os, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

from tensorflow import keras
import segmentation_models as sm
from tensorflow.keras.optimizers import Adam

sm.set_framework('tf.keras')

def lr_schedule():
    def lrs(epoch):
        if epoch >= 0: lr = 0.001
        if epoch >= 1: lr = 0.0001
        if epoch >= 10: lr = 0.00005
        return lr
    return keras.callbacks.LearningRateScheduler(lrs, verbose=True)


def get_segmentation_model(img_size):
    model = sm.Unet('mobilenetv2',
                    input_shape=(img_size, img_size, 3),
                    classes=1,
                    encoder_freeze=False,
                    activation='sigmoid',
                    decoder_filters=(128, 64, 32, 32, 16),
                    encoder_weights='imagenet')

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=sm.losses.bce_jaccard_loss,
                  metrics=[sm.metrics.iou_score])

    return model