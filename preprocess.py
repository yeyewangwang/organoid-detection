#File for preprocessing the organoid images

from skimage import io, img_as_float32
from skimage.transform import resize

from hyperparameters import img_size
import tensorflow as tf

def resize(image):
    return resize(image, (img_size, img_size))

# TODO: rotate
def get_data(self, path, augment):
    if augment:
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocess_fn,
            rotation_range=5,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True,
            brightness_range=(-0.05, 0.05),
            zoom_range=0.1
        )
    else:
        data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=self.preprocess_fn)
    return data_gen.flow_from_directory(path)

