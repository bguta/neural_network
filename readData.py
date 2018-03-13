import h5py as h5
import numpy as np

setName = "basketball"
file1 = "object_nn\\" + setName + "_train.h5"
file2 = "object_nn\\" + setName + "_test.h5"
file3 = "object_nn\\" + setName + "2_train.h5"
file4 = "object_nn\\" + setName + "2_test.h5"

picSize = 28


def read(choice=1):
    """ Read the file choice 1 or 3 is a
    training set, 2 or 4 is a tesing set.
    """
    if choice == 1:
        f = h5.File(file1, "r")
    elif choice == 2:
        f = h5.File(file2, "r")
    elif choice == 3:
        f = h5.File(file3, "r")
    else:
        f = h5.File(file4, "r")

    data = f[setName]

    dataAr = []
    # read the data
    if choice == 1 or choice == 3:
        for i in range(len(data)):
            pic = data[i].reshape(picSize, picSize)  # resize the data
            pSet = []
            # look through the pixels in an individual drawing
            for j in range(len(pic)):
                for pixel in pic[j]:  # per line of pixels
                    pSet.append(pixel)

            # this set contains the whole pixel array size picSize * picSize
            dataAr.append(pSet)
    else:
        dataAr = data
    f.close()
    return dataAr
