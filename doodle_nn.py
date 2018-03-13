# import readData as rd will use later
import model as md
import numpy as np

inputSize = 28 * 28  # the pixels space
outputSize = 3  # the number of choices for objects


def main():
    """ Make the doodle neural net."""
    # bballTrain1 = rd.read()
    Set = np.load("object_nn\\sBasketball.npy")
    data = []

    for i in range(len(Set)):
        data.append(list(Set[i]))

    assert (len(data) != 0)
    assert len(data[0]) == inputSize
    """
    bballTrain2 = rd.read(choice=3)
    assert(len(bballTrain2) != 0 and len(bballTrain2[0]) == inputSize)
    bballTest1 = rd.read(choice=2)
    assert(len(bballTrain2) != 0 and len(bballTrain2[0]) == inputSize)
    bballTest2 = rd.read(choice=4)
    assert(len(bballTrain2) != 0 and len(bballTrain2[0]) == inputSize)
    """

    composition = [inputSize, 500, 300, 200, 100,
                   outputSize]  # the network composition
    md.Network.eta = 0.006
    md.Network.alpha = 0.15

    nn = md.Network(composition)

    goal = [[1, 0, 0]] * len(data)  # [[1,0,0]] means basketball

    # train the network
    train(data, goal, nn)


def train(data, goal, net):
    print("Starting to train...")
    prevE = 0
    i = 0
    while True:

        # input("Enter q to quit else press c: ")
        # start first round of training
        err = 0  # the incured error
        print("Starting epoch " + str(i))

        for j in range(len(data) // 100):
            net.setInput(data[j])

            # feed it through
            net.feedForward()

            # goal is pushed into the outputs neurons
            net.backPropagate(goal[j])

            err += net.getError(goal[j])

        print("\nError: " + str(err))
        print("Change in error: " + str(err - prevE))
        prevE = err
        i += 1

        if err <= 0.01:
            print("Done")
            input()
            break


if __name__ == "__main__":
    main()
