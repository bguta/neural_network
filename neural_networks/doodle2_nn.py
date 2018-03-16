import numpy as np
from PIL import Image as im
import random
import sys
sys.path.append('../')
print(sys.path[-1])
from nn_models import model2 as md


inputSize = 28 * 28  # the pixels space
outputSize = 3  # the number of choices for objects

# labels for the data
BALL = 0
LIGHTBULB = 1
SUN = 2

imgs = ["data/sun.png", "data/bball.png", "data/LightBulb.png"]


def makeData(test=False):
    """ Make the data and return a dict that contains the data and goal."""

    # open the files, I already made them smaller; they contain 3000 pics each
    Ball = np.load("../data/sBasketball.npy")
    LightBulb = np.load("../data/sLight_bulb.npy")
    Sun = np.load("../data/sSun.npy")
    tBall = np.load("../data/sBall_test.npy")
    tLightBulb = np.load("../data/sLight_bulb_test.npy")
    tSun = np.load("../data/sSun_test.npy")

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

    composition = [inputSize, 70, 70, 70, 70, 70,
                   outputSize]  # the network composition

    nn = md.Network(composition)
    nn.eta = 0.0001
    print("LEARNING RATE: " + str(nn.eta) + "\n")

    # train the network
    test(trainingSet["tData"], trainingSet["tGoal"], nn)
    train(trainingSet["data"], trainingSet["goal"], nn)
    test(trainingSet["tData"], trainingSet["tGoal"], nn)

    while True:

        for p in imgs:
            pic = im.open(p).convert("LA")
            pic.load()
            Data = np.asarray(pic, dtype="float32")

            pixels = []
            for i in range(len(Data)):
                for j in range(len(Data[i])):
                    group = list(Data[i][j])
                    # avg = (group[0] + group[1]) // len(group)

                    pixels.append(group[0] / 255.0)

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

            print("EXPECTED: " + p)

            print("")

        input("DONE")
        break


def train(data, goal, net, numEpochs=100):
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
        print("\nEnd of epoch: " + str(i))
        print("\nError: " + str(err))
        print("Change in error: " + str(dE))

        if(err <= 1500):
            break

        if dE >= 100:
            pass
            # net.eta = net.eta = random.uniform(0.000001, net.eta)
            # print("LEARNING RATE: " + str(net.eta) + "\n")
        prevE = err
        i += 1

    # if err <= 1:
    print("Done Training")
    # break
    """

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
