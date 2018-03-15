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


def makeData(test=False):
    """ Make the data and return a dict that contains the data and goal."""

    # open the files, I already made them smaller; they contain 3000 pics each
    Ball = np.load("object_nn\sBasketball.npy")
    LightBulb = np.load("object_nn\sLight_bulb.npy")
    Sun = np.load("object_nn\sSun.npy")
    tBall = np.load("object_nn\sBall_test.npy")
    tLightBulb = np.load("object_nn\sLight_bulb_test.npy")
    tSun = np.load("object_nn\sSun_test.npy")

    data = []
    tData = []  # for the test
    goal = []  # desired output for each pic
    tGoal = []  # for the test

    if test:
        for i in range(len(tBall)):
            tData.append(list(tBall[i]))
            tData[-1].append(BALL)  # add tjhe label to the end of the img

        for i in range(len(tLightBulb)):
            tData.append(list(tLightBulb[i]))
            tData[-1].append(LIGHTBULB)  # add tjhe label to the end of the img
        for i in range(len(tSun)):
            tData.append(list(tSun[i]))
            tData[-1].append(SUN)  # add tjhe label to the end of the img

        random.shuffle(tData)

        for pic in tData:
            # the last index tells us what the image is i.e the label
            OUTPUT = pic[-1]

            if OUTPUT == BALL:
                tGoal.append([1.0, 0.0, 0.0])
            elif OUTPUT == LIGHTBULB:
                tGoal.append([0.0, 1.0, 0.0])
            elif OUTPUT == SUN:
                tGoal.append([0.0, 0.0, 1.0])
            else:
                raise ValueError("We did not get a valid image label")

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

    return {"data": data, "goal": goal, "tData": tData, "tGoal": tGoal}


def main():
    """ Make the doodle neural net."""

    # get the data
    trainingSet = makeData(test=True)

    composition = [inputSize, 30,
                   outputSize]  # the network composition
    md.Network.eta = random.uniform(0.000001, 0.1)
    print("eta: " + str(md.Network.eta))
    md.Network.alpha = 0.1

    nn = md.Network(composition)

    # train the network
    test(trainingSet["tData"], trainingSet["tGoal"], nn)
    train(trainingSet["data"], trainingSet["goal"], nn)
    test(trainingSet["tData"], trainingSet["tGoal"], nn)


def train(data, goal, net, numEpochs=1):
    print("Starting to train...")
    prevE = 0
    i = 0
    # while True:

    # input("Enter q to quit else press c: ")
    for epochs in range(numEpochs):
        err = 0  # the incured error
        print("Starting epoch " + str(i))

        for j in range(len(data) // 3):
            net.setInput(data[j][:-1])  # everythin except the label

            # feed it through
            net.feedForward()

            # goal is pushed into the outputs neurons
            net.backPropagate(goal[j])

            err += net.getError(goal[j])
        print("\n End of epoch: " + str(i))
        print("\nError: " + str(err))
        print("Change in error: " + str(err - prevE))

        if(err - prevE == 0.0):
            break
        prevE = err
        i += 1

    # if err <= 1:
    print("Done Training")
    # break
    """
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
    """


def test(data, goal, net):
    """ Test the model. """

    correct = 0
    print("Testing!")
    for i in range(len(data)):
        net.setInput(data[i][:-1])  # everything except label
        net.feedForward()
        classification = np.argmax(net.getResults())
        answer = np.argmax(goal[i])
        if(answer == classification):
            correct += 1

    print("\n" + str(correct * 100.0 / len(data)) + " %")


if __name__ == "__main__":
    main()
