from pathlib import Path
import cv2
from tensorflow import keras
import numpy as np


data_path = Path('../data/supervisely_segmentation/').expanduser()

images_dir = data_path/'img'
masks_dir = data_path/'masks'
saved_model_dir = "../weights/saved_model/"

def get_segmentation_model_path(size):
    return '../weights/unet_with_wh_{size}.hdf5'.format(size=size)

def load_data(imdir, maskdir):
    image_paths = imdir.glob("*.*")
    data = []
    for im_path in image_paths:
        mask_path = maskdir/(im_path.stem + ".png")
        if mask_path.exists():
            data.append((im_path, mask_path))
    return data

def get_train_data():
    return load_data(images_dir, masks_dir)

class Display:
    def __init__(self, maxsize, time):
        self.maxsize = maxsize
        self.time = time
        self.mode = True
        self.writers = {}

    def getimage(self, image):
        size_ratio = self.maxsize / max(image.shape[0], image.shape[1])
        new_size = int(image.shape[1] * size_ratio), int(image.shape[0] * size_ratio)
        return cv2.resize(image, new_size)

    def show(self, image, name, time=None):
        if not self.mode:
            return

        time = time if time is not None else self.time
        cv2.imshow(name, self.getimage(image))
        cv2.waitKey(time)
        return 0

    def save(self, image, name):
        image = self.getimage(image)
        if name not in self.writers:
            writer = cv2.VideoWriter(name,
                                     cv2.VideoWriter_fourcc('M','J','P','G'),
                                     30, (image.shape[1], image.shape[0]))
            self.writers[name] = writer

        self.writers[name].write(image)

    def off(self):
        self.mode = False

    def on(self):
        self.mode = True


class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, model, samples,  img_size, d_size, d_time, frequency=1):
        self.model = model
        self.samples = list(samples)
        self.img_size = img_size
        self.display =  Display(d_size, d_time)
        self.display.on()
        self.callcount = 0
        self.frequency = frequency

    def draw(self, number):
        outimages = []
        images, masks = self.samples
        for index, item in enumerate(images):
            image, mask = images[index], masks[index]
            pred_mask = self.model.predict(np.expand_dims((image), axis=0))[0]
            pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
            outimage = np.hstack((image, pred_mask, mask))
            outimage = np.asarray(outimage*255, dtype=np.uint8)
            outimages.append(outimage)

        outimage = np.vstack(outimages)
        cv2.putText(outimage, str(number), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        self.display.show(outimage, "zorro")

    def on_batch_end(self, batch, logs=None):
        if batch % self.frequency == 0:
            self.draw(batch)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, batch_size, img_size, shuffle=True):
        self.batch_size = batch_size
        self.img_size = img_size
        self.data = data
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        imgb = np.zeros((self.batch_size, self.img_size, self.img_size, 3), dtype=np.float32)
        maskb = np.zeros((self.batch_size, self.img_size, self.img_size, 3), dtype=np.float32)
        for index, items in enumerate(batch):
            imgpath, maskpath = items
            img = self.get_image(imgpath)
            mask = self.get_image(maskpath)
            mask[mask != 0] = 1.0
            imgb[index] = img
            maskb[index] = mask

        return imgb, maskb

    def get_image(self, path):
        img = cv2.imread(str(path))
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.
        return img

    def sample(self):
        return self[0]

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.data)
