#File for preprocessing the organoid images

from skimage import io, img_as_float32
from skimage.transform import resize

from hyperparameters import img_size

def resize(image):
    return resize(image, (img_size, img_size))

