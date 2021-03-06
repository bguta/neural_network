import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import random
import sys
import math as mt
from functools import reduce
sys.path.insert(0, "../")
from nn_models import model2 as md
from data import formatImage
import time

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

answers = []

for i in range(outputSize):
    answers.append([0] * i + [1] + [0] * (outputSize - 1 - i))


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
    """Make the data and return a dict that contains the data and goal."""
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
    """
    size = reduce(lambda x, y: x + len(y), dTrain, 0)
    print(str(size) + " Training images")
    temp = size
    size += reduce(lambda x, y: x + len(y), dTest, 0)
    print(str(size - temp) + " Testing images")
    print(str(size) + " Total images")
    """

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

lr = 0.01
# random.uniform(0.0001, 100)
eps = 50


def main():
    """Make the doodle neural net."""
    # t_in = time.time()
    # get the data
    trainingSet = makeData(test=True, useBigData=False)
    # print("Time to load data (sec): " + str(time.time() - t_in))

    composition = [inputSize, 100, 20, 10,
                   outputSize]  # the network composition

    nn = md.Network(composition)
    nn.eta = lr
    print("Learning rate: " + str(nn.eta))
    epcs = eps

    # print("LEARNING RATE: " + str(nn.eta) + "\n")

    # do a pre test for the network
    t_in = time.time()
    test(trainingSet["tData"], trainingSet["tGoal"], nn)
    print("Time to test network (sec): " + str(time.time() - t_in) + "\n")

    # train the network
    t_in = time.time()
    train(trainingSet["data"], trainingSet["goal"], nn, numEpochs=epcs,
          plot=True)
    print("Time to train network (sec): " + str(time.time() - t_in))

    # test again
    test(trainingSet["tData"], trainingSet["tGoal"], nn)

    # print("\nTesting created images...")
    # print("labels: " + str(labels) + "\n")
    for p in imgs:
        testImage(p, nn)

    while True:

        imgName = input(
            "please enter the file path of the png pic to test (enter q to quit or s to save): ")
        if imgName == "q":
            break
        if imgName == "s":
            name = input("please enter the name of the file (ex. net1): ")
            nn.save(name)
            print("SAVED")
            break
        testImage(imgName, nn)


def train(data, goal, net, numEpochs=100, plot=True):
    print("Starting to train...")

    prevE = 0
    i = 0

    # plt.ion()  # start the graph
    """
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    fig.suptitle('Error Plot')
    plt.xlabel('Epoch Number')
    plt.ylabel('Error')
    """
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        fig.set_size_inches(10, 10)
        ax1.set_title("RMS Error")
        ax1.set(ylabel="Error")
        ax2.set_title("Learning Rate")
        ax2.set(ylabel="Rate")
        plt.xlabel("Epoch number")
        plt.tight_layout()
        # plt.draw()
        animate.FuncAnimation(fig, addPoint)  # animate the function

    xs, y1 = [], []  # the points
    y2 = []
    decay = net.eta / numEpochs
    load = len(data)
    for epochs in range(numEpochs):
        xs.append(epochs + 1)

        err = 0  # the incured error
        te_in = time.time()
        # print("Starting epoch " + str(i))
        print("Epoch (" + str(epochs + 1) + "/" + str(numEpochs) + ")")
        for j in range(load):
            """
            net.setInput(data[j][:-1])  # everythin except the label

            # feed it through
            net.feedForward()

            # goal is pushed into the outputs neurons
            net.backPropagate(goal[j])
            """

            prograssBar(j + 1, load)
            err += net.train(data[j][:-1], goal[j])
        print("\n")
        dE = err - prevE  # change in error
        print("Time to go through epoch #" + str(i + 1) +
              " (sec): " + str(time.time() - te_in))

        if plot:
            y1.append(err)  # the y point
            y2.append(net.eta)
            addPoint(xs, y1, ax1, shape="-")
            addPoint(xs, y2, ax2, colour="b")
            plt.draw()
            plt.pause(0.0001)

        # print("End of epoch: " + str(i))
        print("Learning rate: " + str(net.eta))
        print("Error: " + str(err))
        print("Change in error: " + str(dE) + "\n")
        # net.eta = 100 / (epochs + 1)  # decrease the learning rate
        # changeLearningRate(net, epochs, decay)  # change the learning rate

        if err <= 200:
            break

        if not (dE <= 0.001 and dE >= -0.001):
            if err >= 200:
                searchThanConv(net, epochs)
        #     print("END")
        #     break
            # net.eta = random.uniform(50, 100)
            # net.eta = net.eta = random.uniform(0.000001, net.eta)
        prevE = err
        i += 1

    # if err <= 1:
    # print("Done training\n")
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
        try:
            if img in imgs:
                pic = image.imread(img)
            else:
                formatImage.format(img)
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


def addPoint(xs, ys, axis, colour="r", shape="o"):
    """
    animate the plot of the error

    @param xs a list of x points
    @param ys a list of y points
    @param axis i.e suplot
    """
    axis.plot(xs, ys, colour + shape)
    return True


def prograssBar(val, final):
    """
    Show the prograss.

    @param val
    the current value of the prograss (you should increase this yourself)

    @param final
    the final goal
    """
    maxlen = 50
    step = final // maxlen

    print("\r[ " + "#" * (val // step) + " ] " +
          str(int(val * 100.0 / final)) + "% ", end="")


def changeLearningRate(net, epoch, decay):
    """
    @param net
    the neural network

    @param epochs
    the epoch number that training is being run for

    @param decay
    the decay calculated as the original lr devided by the total epochs

    @changes
    the learning rate
    """
    net.eta = net.eta * 1 / (1 + decay * epoch)


def searchThanConv(net, epoch, eta=lr, searchE=int(eps * 0.8), alpha=10):
    """
    search then converge- (STC) learning rate
    schedules (Darken and Moody, 1990b, 1991)

    This is constant at eta for a given number of epochs (searchE)
    than it begins to decrease

    @param alpha - a constatnt
    @pararm eta - the original learning rate
    @param searchE - the number of epochs to maintain eta for
    @param net - the neural net
    @param epoch - the epoch number

    visit here for more info: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.2884&rep=rep1&type=pdf
    """
    net.eta = eta * (1 + (alpha / eta) * (epoch / searchE)) / \
        (1 + (alpha / eta) * (epoch / searchE) + (epoch**2 / searchE))


if __name__ == "__main__":
    main()
