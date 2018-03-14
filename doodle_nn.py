# import readData as rd will use later
import model as md
import numpy as np
from PIL import Image as im
import random

inputSize = 28 * 28  # the pixels space
outputSize = 3  # the number of choices for objects


# labels for the data
BALL = 0
LIGHTBULB = 1
SUN = 2


def makeData():
    """ Make the data and return a dict that contains the data and goal."""

    # open the files, I already made them smaller; they contain 3000 pics each
    Ball = np.load("object_nn\sBasketball.npy")
    LightBulb = np.load("object_nn\sLight_bulb.npy")
    Sun = np.load("object_nn\sSun.npy")

    data = []

    for i in range(len(Ball)):
        data.append(list(Ball[i]))
        data[-1].append(BALL)  # add tjhe label to the end of the img

    for i in range(len(LightBulb)):
        data.append(list(LightBulb[i]))
        data[-1].append(LIGHTBULB)  # add tjhe label to the end of the img
    for i in range(len(Sun)):
        data.append(list(Sun[i]))
        data[-1].append(SUN)  # add tjhe label to the end of the img

    random.shuffle(data)

    assert (len(data) != 0)
    assert len(data[0]) - 1 == inputSize  # minus the label

    goal = []  # desired output for each pic
    for pic in data:
        # the last index tells us what the image is i.e the label
        OUTPUT = pic[-1]

        if OUTPUT == BALL:
            goal.append([1.0, 0.0, 0.0])
        elif OUTPUT == LIGHTBULB:
            goal.append([0.0, 1.0, 0.0])
        elif OUTPUT == SUN:
            goal.append([0.0, 0.0, 1.0])
        else:
            raise ValueError("We did not get a valid image label")

    return {"data": data, "goal": goal}


def main():
    """ Make the doodle neural net."""
    # bballTrain1 = rd.read()

    trainingSet = makeData()
    """
    bballTrain2 = rd.read(choice=3)
    assert(len(bballTrain2) != 0 and len(bballTrain2[0]) == inputSize)
    bballTest1 = rd.read(choice=2)
    assert(len(bballTrain2) != 0 and len(bballTrain2[0]) == inputSize)
    bballTest2 = rd.read(choice=4)
    assert(len(bballTrain2) != 0 and len(bballTrain2[0]) == inputSize)
    """

    composition = [inputSize, 30,
                   outputSize]  # the network composition
    md.Network.eta = 0.1
    md.Network.alpha = 1

    nn = md.Network(composition)

    # train the network
    train(trainingSet["data"], trainingSet["goal"], nn)


def train(data, goal, net):
    print("Starting to train...")
    prevE = 0
    i = 0
    # while True:

    # input("Enter q to quit else press c: ")
    # start first round of training
    err = 0  # the incured error
    print("Starting epoch " + str(i))

    for j in range(len(data) // 4):
        net.setInput(data[j][:-1])  # everythin except the label

        # feed it through
        net.feedForward()

        # goal is pushed into the outputs neurons
        net.backPropagate(goal[j])

        err += net.getError(goal[j])
    print("\n End of epoch: " + str(i))
    print("\nError: " + str(err))
    print("Change in error: " + str(err - prevE))
    # if(err - prevE == 0.0):
    # break
    prevE = err
    i += 1

    # if err <= 1:
    #  print("Done")
    # break

    while True:
        pic = im.open("bball.png").convert("LA")
        pic.load()
        Data = np.asarray(pic, dtype="float32")

        pixels = []
        for i in range(len(Data)):
            for j in range(len(Data[i])):
                group = list(Data[i][j])
                # avg = (group[0] + group[1]) // len(group)

                pixels.append(group[0] / 255.0)

        net.setInput(pixels)
        net.feedForward()
        print(str(net.getResults()))
        input()
        break


if __name__ == "__main__":
    main()
