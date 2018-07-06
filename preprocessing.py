import numpy as np

root = "./OpportunityUCIDataset/dataset/S"
for sub in [1]:#[1,2,3,4]:
    for sess in [1]:#[1,2,3,4,5,6]:
        if sess < 6:
            filename = root + str(sub) + "-ADL" + str(sess) + ".dat"
            print("importing from: ", filename)
            data = np.genfromtxt(filename,
                                 usecols=([1:46]))
            print(data.shape)

        else:
            filename = root + str(sub) + "-Drill.dat"
            print("importing from: ", filename)
