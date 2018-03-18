import numpy as np
import sys
path = "../../data/"


def main():
    fileName = input("Please tell me the file name (ex: basketball.npy): ")
    percentTrain = input(
        "what percent is for training (for example 80 for 80%): ")

    data = np.load(path + fileName)
    data = data.astype("float32") / 255  # make small

    seed = np.random.randint(1, 10e6)
    np.random.seed(seed)
    np.random.shuffle(data)

    split = len(data) * (float(percentTrain) / 100)
    split = int(split)

    d_train = data[:split]
    d_test = data[split:]

    name = fileName.split(".npy")

    np.save(name[0] + "train", d_train)
    np.save(name[0] + "test", d_test)


if __name__ == "__main__":
    main()
