import numpy as np
from PIL import Image as Im
import PIL.ImageOps as imo
import matplotlib.image as image
from scipy.misc import imsave

""" save a doodle made in paint as a png in data/ then
    run this on the image to format it. This overrides the image
    so make sure to have a backup.
"""


def main():
    imName = input("Please enter the name of the image (ex: sun.png): ")

    im = Im.open(imName)
    im = im.resize((28, 28), Im.ANTIALIAS)

    im = im.convert('L')
    im = imo.invert(im)
    im = im.convert('RGB')
    im.save(imName)

    pic = image.imread(imName)
    gpic = rgb2gray(pic)
    imsave(imName, gpic)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def format(imName, invert=True):
    """
    format a given image (png made in paint) to make it a 28 by 28 greyscale image
    """
    im = Im.open(imName)
    im = im.resize((28, 28), Im.ANTIALIAS)

    if invert:
        im = im.convert('L')
        im = imo.invert(im)

    im = im.convert('RGB')
    im.save(imName)

    pic = image.imread(imName)
    gpic = rgb2gray(pic)
    imsave(imName, gpic)


if __name__ == "__main__":
    main()
