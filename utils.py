import os
import random
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

windowSize = 100
filePath = "/content/drive/My Drive/sleepStageData"

def rescaleArray(X):
    """
    Rescale an array.
    Map the minimum element of the array to −5 and the maximum to +5.
    """
    X = X / 20

    X = np.clip(X, -5, 5)
    return X

def generator(dictFiles):
    """
    A generator is used to deal with the large dataset.
    The Keras model will be fed data from this generator.
    """
    while True:
        # choose a random file
        recordName = random.choice(list(dictFiles.keys()))
        batchData = dictFiles[recordName]

        # get X and Y values from the .npz file
        allRows = batchData['x']
        allLabels = batchData['y']

        for j in range(10):
            startIndex = random.choice(range(allRows.shape[0] - windowSize))

            X = allRows[startIndex:startIndex + windowSize]
            Y = allLabels[startIndex:startIndex + windowSize]

            # expand arrays
            X = np.expand_dims(X, 0)
            Y = np.expand_dims(Y, -1)
            Y = np.expand_dims(Y, 0)

            X = rescaleArray(X)

            yield X, Y

def chunker(sequence, size=windowSize):
    """
    Split data into chunks. Used for testing the model.
    """
    for position in range(0, len(sequence), size):
        return sequence[position:position + size]

def getSets(path=filePath):
    """
    Split the dataset into dictionaries for training, testing, and validation.
    """
    files = sorted(glob(os.path.join(filePath, "*.npz")))
    IDs = sorted(list(set([j.split("/")[-1][:5] for j in files])))

    # use the same random state every time
    trainIDs, testIDs = train_test_split(IDs, test_size=0.15, random_state=7)
    
    trainValidation, test = [j for j in files if j.split("/")[-1][:5] in trainIDs], [j for j in files if j.split("/")[-1][:5] in testIDs]
    train, validation = train_test_split(trainValidation, test_size=0.1, random_state=18)

    # dictionaries used for training
    valDict = {k: np.load(k) for k in validation}
    trainDict = {k: np.load(k) for k in train}
    testDict = {k: np.load(k) for k in test}

    return trainDict, testDict, valDict