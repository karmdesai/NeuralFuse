from utils import chunker
from utils import getSets
from utils import rescaleArray

from sklearn.metrics import accuracy_score
from keras.models import load_model
import numpy as np

# load testing data (a random state was used to ensure that it was shuffled in the same way)
model = load_model("/content/drive/My Drive/sleepStageData/cnnFinal.h5")
trainDict, testDict, valDict = getSets()

# store prediction labels and actual labels in two separate lists
realLabels = []
modelPredictions = []

for eachKey in testDict:
    """
    Testing data is processed and the model's predictions are stored in a list.
    """
    allRows = testDict[eachKey]['x']

    for batchHyp in chunker(range(allRows.shape[0])):

        X = allRows[min(batchHyp):max(batchHyp) + 1, ...]
        Y = testDict[eachKey]['y'][min(batchHyp):max(batchHyp) + 1]

        X = np.expand_dims(X, 0)

        X = rescaleArray(X)

        yPred = model.predict(X)
        yPred = yPred.argmax(axis=-1).ravel().tolist()

        realLabels += Y.ravel().tolist()
        modelPredictions += yPred

# compare predicted labels to actual labels
modelAccuracy = accuracy_score(realLabels, modelPredictions)
print("Model's Accuracy Score: ", modelAccuracy)