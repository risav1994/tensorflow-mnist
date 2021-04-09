import tensorflow as tf
from random import choice


def get_unique_indices(indx_set, size=64, **kwargs):
    assert len(indx_set) >= size
    unique_indices = []
    for i in range(size):
        unique_indx = choice(indx_set)
        indx_set.remove(unique_indx)
        unique_indices.append(unique_indx)
    return unique_indices


def getGenerator(x, y, batchSize, **kwargs):
    indx_set = list(range(len(x)))
    while True:
        if len(indx_set) <= batchSize:
            yield (x[indx_set], y[indx_set])
            indx_set = list(range(len(x)))
            continue
        unique_indices = get_unique_indices(indx_set, batchSize)
        yield (x[unique_indices], y[unique_indices])


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

