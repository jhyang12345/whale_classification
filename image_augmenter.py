import os, sys
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from imgaug import augmenters as iaa
from PIL import Image

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0, 0.8)),
        iaa.Affine(
            scale=(1.0, 1.3),
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-30, 30),
        ),
        iaa.Add((-45, 45)),
    ],
    random_order=True)

# save augmented versions of the given image in a parallel directory called augmented
def augment_image_save(full_path, times=9):
    file_path = os.path.dirname(full_path)
    file_name = os.path.basename(full_path)
    file_head = file_name.split(".")[0]
    original = load_img(full_path)
    ret = [load_img(full_path) for _ in range(times)]
    ret = [img_to_array(item) for item in ret]
    images = seq.augment_images(ret)
    new_path = os.path.join(file_path, "..")
    new_path = os.path.join(new_path, "augmented")
    for i, image in enumerate(images):
        new_name = "{}_{}".format(file_head, i)
        new_name = new_name + ".jpg"
        im = Image.fromarray(np.uint8(image))
        im.save(os.path.join(new_path, new_name))
    im = Image.fromarray(np.uint8(original))
    im.save(os.path.join(new_path, file_name))

def filter_images_in_path(filter_path, min_size=80):
    images = os.listdir(filter_path)
    ret = []
    for image in images:
        full_path = os.path.join(filter_path, image)
        im = Image.open(full_path)
        if(im.size[0] > min_size):
            ret.append(image)
    return ret

def augment_directory(directory):
    images = filter_images_in_path(directory)
    for image in images:
        full_path = os.path.join(directory, image)
        augment_image_save(full_path)

if __name__ == '__main__':
    augment_directory("train")