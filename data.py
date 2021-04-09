import tensorflow as tf
from random import choice


def getUniqueIndices(indxSet, size=64, **kwargs):
    assert len(indxSet) >= size
    uniqueIndices = []
    for i in range(size):
        uniqueIndx = choice(indxSet)
        indxSet.remove(uniqueIndx)
        uniqueIndices.append(uniqueIndx)
    return uniqueIndices


def getGenerator(x, y, batchSize, **kwargs):
    indxSet = list(range(len(x)))
    while True:
        if len(indxSet) <= batchSize:
            yield (x[indxSet], y[indxSet])
            indxSet = list(range(len(x)))
            continue
        uniqueIndices = getUniqueIndices(indxSet, batchSize)
        yield (x[uniqueIndices], y[uniqueIndices])


def getDataGenerator(**kwargs):
    type_ = kwargs.get("type_", "train")
    batchSize = kwargs.get("batchSize", 64)
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    xTrain = xTrain.astype("float32") / 255
    xTest = xTest.astype("float32") / 255
    if type_ == "train":
        return getGenerator(xTrain, yTrain, batchSize)
    if type_ == "test":
        return getGenerator(xTest, yTest, batchSize)

