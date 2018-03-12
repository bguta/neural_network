import h5py as h5
import numpy as np

setName = "basketball"
file = "object_nn\\" + setName + "_train.h5"

picSize = 28


def read():
    f = h5.File(file, "r")
    data = f[setName]

    dataAr = []

    for i in range(len(data)):
        pic = data[i].reshape(picSize, picSize)

        for j in range(picSize):
            Set = []
            for pixel in pic[j]:
                Set.append(pixel)
            dataAr.append(Set)
    f.close()
    return dataAr
