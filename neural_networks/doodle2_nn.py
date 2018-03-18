import numpy as np
import matplotlib.image as image
import random
import sys
sys.path.append('../')
from nn_models import model2 as md
import time

inputSize = 28 * 28  # the pixels space
outputSize = 5  # the number of choices for objects

# labels for the data
BALL = 0
LIGHTBULB = 1
SUN = 2
CLOUD = 3
EYE = 4


imgs = [
    "../data/sun.png",
    "../data/bball.png",
    "../data/LightBulb.png",
    "../data/basketball.png"]


def makeData(test=False, useBigData=False):
    """ Make the data and return a dict that contains the data and goal."""

    print("Importing Data...")

    # open the files, I already made them smaller; they contain 3000 pics each
    if useBigData:
        Ball = np.load("../data/basketballtrain.npy")
        LightBulb = np.load("../data/light_bulbtrain.npy")
        Sun = np.load("../data/suntrain.npy")
        Cloud = np.load("../data/cloudtrain.npy")
        Eye = np.load("../data/eyetrain.npy")

        if test:  # load the tests
            tBall = np.load("../data/basketballtest.npy")
            tLightBulb = np.load("../data/light_bulbtest.npy")
            tSun = np.load("../data/suntest.npy")
            tCloud = np.load("../data/cloudtest.npy")
            tEye = np.load("../data/eyetest.npy")
    else:
        Ball = np.load("../data/sBasketball.npy")
        LightBulb = np.load("../data/sLight_bulb.npy")
        Sun = np.load("../data/sSun.npy")
        Cloud = np.load("../data/sCloud_train.npy")
        Eye = np.load("../data/sEye_train.npy")

        if test:  # load the test
            tBall = np.load("../data/sBall_test.npy")
            tLightBulb = np.load("../data/sLight_bulb_test.npy")
            tSun = np.load("../data/sSun_test.npy")
            tCloud = np.load("../data/sCloud_test.npy")
            tEye = np.load("../data/sEye_test.npy")

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

        for i in range(len(tCloud)):
            tData.append(list(tCloud[i]))
            tData[-1].append(CLOUD)  # add tjhe label to the end of the img

        for i in range(len(tEye)):
            tData.append(list(tEye[i]))
            tData[-1].append(EYE)  # add tjhe label to the end of the img

        random.shuffle(tData)

        for pic in tData:
            # the last index tells us what the image is i.e the label
            OUTPUT = pic[-1]

            if OUTPUT == BALL:
                tGoal.append([1.0, 0.0, 0.0, 0.0, 0.0])
            elif OUTPUT == LIGHTBULB:
                tGoal.append([0.0, 1.0, 0.0, 0.0, 0.0])
            elif OUTPUT == SUN:
                tGoal.append([0.0, 0.0, 1.0, 0.0, 0.0])
            elif OUTPUT == CLOUD:
                tGoal.append([0.0, 0.0, 0.0, 1.0, 0.0])
            elif OUTPUT == EYE:
                tGoal.append([0.0, 0.0, 0.0, 0.0, 1.0])
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
        data[-1].append(SUN)  # add the label to the end of the img

    for i in range(len(Cloud)):
        data.append(list(Cloud[i]))
        data[-1].append(CLOUD)  # add tjhe label to the end of the img

    for i in range(len(Eye)):
        data.append(list(Eye[i]))
        data[-1].append(EYE)  # add tjhe label to the end of the img

    random.shuffle(data)

    assert (len(data) != 0)
    assert len(data[0]) - 1 == inputSize  # minus the label

    for pic in data:
        # the last index tells us what the image is i.e the label
        OUTPUT = pic[-1]

        if OUTPUT == BALL:
            goal.append([1.0, 0.0, 0.0, 0.0, 0.0])
        elif OUTPUT == LIGHTBULB:
            goal.append([0.0, 1.0, 0.0, 0.0, 0.0])
        elif OUTPUT == SUN:
            goal.append([0.0, 0.0, 1.0, 0.0, 0.0])
        elif OUTPUT == CLOUD:
            goal.append([0.0, 0.0, 0.0, 1.0, 0.0])
        elif OUTPUT == EYE:
            goal.append([0.0, 0.0, 0.0, 0.0, 1.0])
        else:
            raise ValueError("We did not get a valid image label")

    return {"data": data, "goal": goal, "tData": tData, "tGoal": tGoal}


def main():
    """ Make the doodle neural net."""
    t_in = time.time()
    # get the data
    trainingSet = makeData(test=True, useBigData=False)

    composition = [inputSize, 100,
                   outputSize]  # the network composition

    nn = md.Network(composition)
    nn.eta = 0.0001
    print("LEARNING RATE: " + str(nn.eta) + "\n")

    # train the network
    test(trainingSet["tData"], trainingSet["tGoal"], nn)
    train(trainingSet["data"], trainingSet["goal"], nn)
    test(trainingSet["tData"], trainingSet["tGoal"], nn)

    print("time: " + str(time.time() - t_in) + "s \n")
    input()
    """
    for p in imgs:
        testImage(p, nn)

    while True:

        imgName = input(
            "please enter the file path of the formatted 28 by 28 pic: ")
        testImage(imgName, nn)
    """


def train(data, goal, net, numEpochs=1):
    print("Starting to train...")
    prevE = 0
    i = 0
    # while True:

    # input("Enter q to quit else press c: ")
    for epochs in range(numEpochs):
        err = 0  # the incured error
        print("Starting epoch " + str(i))
        for j in range(len(data)):
            net.setInput(data[j][:-1])  # everythin except the label

            # feed it through
            net.feedForward()

            # goal is pushed into the outputs neurons
            net.backPropagate(goal[j])

            err += net.getError(goal[j])

        dE = err - prevE
        print("End of epoch: " + str(i))
        print("Error: " + str(err))
        print("Change in error: " + str(dE) + "\n")

        if err < 200:
            break

        if dE >= 100:
            pass
            # net.eta = net.eta = random.uniform(0.000001, net.eta)
            # print("LEARNING RATE: " + str(net.eta) + "\n")
        prevE = err
        i += 1

    # if err <= 1:
    print("Done training\n")
    # break
    """

    """


def test(data, goal, net):
    """ Test the model. """
    correct = 0
    print("Testing...")
    for i in range(len(data)):
        net.setInput(data[i][:-1])  # everything except label
        net.feedForward()
        classification = np.argmax(net.getResults())
        answer = np.argmax(goal[i])
        if(answer == classification):
            correct += 1

    print("Score: " + str(correct * 100.0 / len(data)) + " %\n")


def testImage(img, nn):
    if ".png" in img:
        pic = image.imread(img)

        pixels = pic.reshape(int(28 * 28), 1)

        nn.setInput(pixels)
        nn.feedForward()
        v = nn.getResults()
        print(str(v))
        ans = np.argmax(v)

        if ans == BALL:
            print("BALL")
        elif ans == LIGHTBULB:
            print("LIGHTBULB")
        elif ans == SUN:
            print("SUN")
        elif ans == CLOUD:
            print("CLOUD")
        elif ans == EYE:
            print("EYE")

        print("EXPECTED: " + img + "\n")


if __name__ == "__main__":
    main()
