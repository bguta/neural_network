import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import random
import sys
import math as mt
sys.path.insert(0, "../")
from nn_models import model2 as md  # noqa
import time  # noqa

inputSize = 28 * 28  # the pixels space
outputSize = 8  # the number of choices for objects

# labels for the data : this order matters!!
BALL = 0
LIGHTBULB = 1
SUN = 2
CLOUD = 3
EYE = 4
BIKE = 5
DOG = 6
FLOWER = 7
labels = ["ball", "lightbulb", "sun", "cloud", "eye", "bike", "dog", "flower"]

answers = [
    [1] + [0] * (outputSize - 1),
    [0] * 1 + [1] + [0] * (outputSize - 2),
    [0] * 2 + [1] + [0] * (outputSize - 3),
    [0] * 3 + [1] + [0] * (outputSize - 4),
    [0] * 4 + [1] + [0] * (outputSize - 5),
    [0] * 5 + [1] + [0] * (outputSize - 6),
    [0] * 6 + [1] + [0] * (outputSize - 7),
    [0] * 7 + [1] + [0] * (outputSize - 8)
]


imgs = [
    "../data/pics/basketball.png",
    "../data/pics/LightBulb.png",
    "../data/pics/sun.png",
    "../data/pics/cloud.png",
    "../data/pics/eye.png"
]
large_Data = [  # the large data; note the order, it is the same as above numbering
    [
        "../data/l/basketballtrain.npy",
        "../data/l/light_bulbtrain.npy",
        "../data/l/suntrain.npy",
        "../data/l/cloudtrain.npy",
        "../data/l/eyetrain.npy",
        "../data/l/bicycletrain.npy",
        "../data/l/dogtrain.npy",
        "../data/l/flowertrain.npy"
    ],
    [
        "../data/l/basketballtest.npy",
        "../data/l/light_bulbtest.npy",
        "../data/l/suntest.npy",
        "../data/l/cloudtest.npy",
        "../data/l/eyetest.npy",
        "../data/l/bicycletest.npy",
        "../data/l/dogtest.npy",
        "../data/l/flowertest.npy"
    ]
]
small_Data = [  # the small data set
    [
        "../data/s/sBallTrain.npy",
        "../data/s/sLightbulbTrain.npy",
        "../data/s/sSunTrain.npy",
        "../data/s/sCloudTrain.npy",
        "../data/s/sEyeTrain.npy",
        "../data/s/sBicycleTrain.npy",
        "../data/s/sDogTrain.npy",
        "../data/s/sFlowerTrain.npy"
    ],
    [
        "../data/s/sBallTest.npy",
        "../data/s/sLightbulbTest.npy",
        "../data/s/sSunTest.npy",
        "../data/s/sCloudTest.npy",
        "../data/s/sEyeTest.npy",
        "../data/s/sBicycleTest.npy",
        "../data/s/sDogTest.npy",
        "../data/s/sFlowerTest.npy"
    ]
]


def makeData(test=False, useBigData=False):
    """ Make the data and return a dict that contains the data and goal."""

    print("Importing Data...")
    dTrain = []
    dTest = []

    # open the files, I already made them smaller; they contain 3000 pics each
    if useBigData:
        for i in range(len(large_Data[0])):
            val = np.load(large_Data[0][i])
            dTrain.append(val)

        if test:  # load the tests
            for i in range(len(large_Data[1])):
                val = np.load(large_Data[1][i])
                dTest.append(val)
    else:
        for i in range(len(small_Data[0])):
            val = np.load(small_Data[0][i])
            dTrain.append(val)

        if test:  # load the test

            for i in range(len(small_Data[1])):
                val = np.load(small_Data[1][i])
                dTest.append(val)

    data = []
    tData = []  # for the test
    goal = []  # desired output for each pic
    tGoal = []  # for the test

    if test:
        for i in range(len(dTest)):
            for vct in dTest[i]:
                vct = np.append(vct, i)  # input the label
                vec = vct.reshape(inputSize + 1, 1)
                tData.append(vec)

        random.shuffle(tData)

        for pic in tData:
            # the last index tells us what the image is i.e the label
            OUTPUT = pic[-1]

            assert OUTPUT < outputSize
            assert OUTPUT >= 0

            tGoal.append(answers[int(OUTPUT)])

    for i in range(len(dTrain)):
        for vct in dTrain[i]:
            vct = np.append(vct, i)
            vec = vct.reshape(inputSize + 1, 1)
            data.append(vec)

    random.shuffle(data)

    assert (len(data) != 0)
    assert len(data[0]) - 1 == inputSize  # minus the label

    for pic in data:
        # the last index tells us what the image is i.e the label
        OUTPUT = pic[-1]

        assert OUTPUT < outputSize
        assert OUTPUT >= 0

        goal.append(answers[int(OUTPUT)])

    return {"data": data, "goal": goal, "tData": tData, "tGoal": tGoal}


def main():
    """ Make the doodle neural net."""
    t_in = time.time()
    # get the data
    trainingSet = makeData(test=True, useBigData=True)

    composition = [inputSize, 100, 10,
                   outputSize]  # the network composition

    nn = md.Network(composition)
    nn.eta = 0.001
    epcs = 5

    print("LEARNING RATE: " + str(nn.eta) + "\n")

    # train the network
    test(trainingSet["tData"], trainingSet["tGoal"], nn)
    train(trainingSet["data"], trainingSet["goal"], nn, numEpochs=epcs)
    test(trainingSet["tData"], trainingSet["tGoal"], nn)

    print("time: " + str(time.time() - t_in) + "s \n")
    print("\nTesting created images...")

    for p in imgs:
        testImage(p, nn)

    while True:

        imgName = input(
            "please enter the file path of the png pic to test (enter q to quit): ")
        if imgName == "q":
            break
        testImage(imgName, nn)


def addPoint(xs, ys, axis):
    """
    animate the plot of the error

    @param xs a list of x points
    @param ys a list of y points
    @param axis i.e suplot
    """
    axis.plot(xs, ys, "ro")


def train(data, goal, net, numEpochs=100):
    print("Starting to train...")

    prevE = 0
    i = 0

    plt.ion()  # start the graph
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    fig.suptitle('Error Plot')
    plt.xlabel('Epoch Number')
    plt.ylabel('Error')
    xs, ys = [], []  # the points

    for epochs in range(numEpochs):
        xs.append(epochs)

        err = 0  # the incured error
        print("Starting epoch " + str(i))
        for j in range(len(data)):
            """
            net.setInput(data[j][:-1])  # everythin except the label

            # feed it through
            net.feedForward()

            # goal is pushed into the outputs neurons
            net.backPropagate(goal[j])
            """
            err += net.train(data[j][:-1], goal[j])

        dE = err - prevE  # change in error

        ys.append(err)  # the y point
        animate.FuncAnimation(fig, addPoint(xs, ys, axis))  # draw the graph
        plt.show()
        plt.pause(0.1)

        print("End of epoch: " + str(i))
        print("Error: " + str(err))
        print("Change in error: " + str(dE) + "\n")

        # net.eta = mt.exp(-(epochs + 1))
        print("LEARNING RATE: " + str(net.eta) + "\n")

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
        """
        net.setInput(data[i][:-1])  # everything except label
        net.feedForward()
        """
        classification = np.argmax(net.test(data[i][:-1]))
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

        label = img.split("/")[-1]
        print(labels[ans] + " ; EXPECTED " + label + "\n")


if __name__ == "__main__":
    main()
