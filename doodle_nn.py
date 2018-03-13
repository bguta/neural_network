import readData as rd
import model as md

inputSize = 28 * 28  # the pixels space
outputSize = 3  # the number of choices for objects


def main():
    """ Make the doodle neural net."""
    bballTrain1 = rd.read()
    assert (len(bballTrain1) != 0)
    assert len(bballTrain1[0]) == inputSize
    """
    bballTrain2 = rd.read(choice=3)
    assert(len(bballTrain2) != 0 and len(bballTrain2[0]) == inputSize)
    bballTest1 = rd.read(choice=2)
    assert(len(bballTrain2) != 0 and len(bballTrain2[0]) == inputSize)
    bballTest2 = rd.read(choice=4)
    assert(len(bballTrain2) != 0 and len(bballTrain2[0]) == inputSize)
    """

    composition = [inputSize, 1000, 1000, 1000,
                   outputSize]  # the network composition
    md.Network.eta = 0.0001
    md.Network.alpha = 0.0001

    nn = md.Network(composition)

    goal = [[1, 0, 0]] * len(bballTrain1)  # [[1,0,0]] means basketball

    # train the network
    train(bballTrain1, goal, nn)


def train(data, goal, net):
    print("Starting to train...")
    while True:

        # input("Enter q to quit else press c: ")
        # start first round of training
        err = 0  # the incured error

        for j in range(len(data) // 1000):
            print("Starting epoch " + str(j))

            net.setInput(data[j])

            # feed it through
            net.feedForward()

            # goal is pushed into the outputs neurons
            net.backPropagate(goal[j])

            err += net.getError(goal[j])

            print("\nError: " + str(err))

        if err <= 0.01:
            print("Done")
            input()
            break


if __name__ == "__main__":
    main()
