import numpy as np
import sys
sys.path.insert(0, "../")
from nn_models import model2 as md
from data import formatImage
import matplotlib.image as image


"""
This initalizes a neural net with a given file and test the file on a given image.
"""
inputSize = 28 * 28  # the pixels space
outputSize = 10  # the number of choices for objects

# labels for the data : this order matters!!
BALL = 0
LIGHTBULB = 1
SUN = 2
CLOUD = 3
EYE = 4
BIKE = 5
DOG = 6
FLOWER = 7
#labels = ["ball", "lightbulb", "sun", "cloud", "eye", "bike", "dog", "flower"]
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

imgs = [
    "../data/pics/basketball.png",
    "../data/pics/LightBulb.png",
    "../data/pics/sun.png",
    "../data/pics/cloud.png",
    "../data/pics/eye.png",
    "../data/pics/bike.png",
    "../data/pics/dog.png",
    "../data/pics/flower.png"
]


def main():

    composition = [inputSize, 100, 10,
                   outputSize]  # the network composition
    nn = md.Network(composition)

    while True:
        name = input(
            "please enter the name of the file to initalize the network with (ex. net1): ")

        try:
            nn.load(name)
            break
        except Exception as e:
            print(e)

    while True:
        imgName = input(
            "please enter the file path of the png pic to test (enter q to quit): ")
        if imgName == "q":
            break

        testImage(imgName, nn)


def testImage(img, nn):
    """
    Test an image with the network
    """
    if ".png" in img:
        try:
            if img in imgs:
                pic = image.imread(img)
            else:
                formatImage.format(img, invert=True)
                pic = image.imread(img)

            pixels = pic.reshape(int(28 * 28), 1)

            v = nn.test(pixels)
            # print(str(v))
            ans = np.argmax(v)

            label = img.split("/")[-1]
            print(labels[ans] + " " + str(v[ans]) +
                  " : EXPECTED " + label + "\n")
        except Exception as e:
            print(e)
            pass

if __name__ == "__main__":
    main()
