import os, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)


from utils import (data_path, Display, get_train_data, DisplayCallback, DataGenerator)
import tensorflow as tf
from common import lr_schedule, get_segmentation_model


def train(output_dir, epochs=1):
    img_size = 128
    batch_size = 4

    model = get_segmentation_model(img_size)
    print (model.summary())

    data = get_train_data()[:500]
    val_split = 0.1
    train_size = int(len(data)*(1 - val_split))
    train_data = data[:train_size]
    val_data = data[train_size:]

    train_generator = DataGenerator(train_data, batch_size, img_size)
    val_generator = DataGenerator(val_data, batch_size, img_size)

    sample_batch = val_generator.sample()
    display_callback = DisplayCallback(model, sample_batch, img_size, 600, 100, frequency=1)

    model_path = output_dir + "/weights.hdf5"
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                       monitor='val_loss',
                                       save_best_only=True,
                                       verbose=1)

    callbacks = [lr_schedule(), cp, display_callback]

    history = model.fit_generator(train_generator,
                        validation_data = val_generator,
                        epochs=epochs,
                        shuffle=True,
                        callbacks=callbacks,
                        verbose=1,
                        use_multiprocessing=True,
                        workers=4)

    model.load_weights(model_path)
    tf.saved_model.save(model, output_dir + "/saved_model/")
    return model
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=None, required=True)

    args = parser.parse_args()
    train(args.outdir)