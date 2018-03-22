import numpy as np


def main():
	name = input("Please input the file name without the .npy (ex: dog): ")
    data = np.load(name + ".npy")
    data = data.astype("float32") / 255
    d_tr = data[:3000]
    d_te = data[3000:3200]
    np.save("s" + name +"Test", d_te)
    np.save("s" + name + "Train", d_tr)
