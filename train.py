import tensorflow as tf
from tqdm import tqdm
from data import getDataGenerator
from model import getModel

dataTrainGen = getDataGenerator()
dataTestGen = getDataGenerator(type_="test", batchSize=10000)


def trainOps(loss, **kwargs):
    optimizer = tf.train.AdamOptimizer(
        name="adamOptimizer",
        learning_rate=5e-5
    )
    trainVars = tf.trainable_variables()
    accumVars = [
        tf.get_variable(
            f"train_var_{idx}",
            initializer=tf.zeros_like(
                trainVar.read_value()
            ),
            trainable=False
        ) for idx, trainVar in enumerate(trainVars)
    ]
    zeroOps = [
        accumVar.assign(
            tf.zeros_like(
                accumVar
            )
        ) for accumVar in accumVars
    ]
    gradVars = optimizer.compute_gradients(loss, trainVars)
    gradVars = [
        (tf.where(tf.is_nan(gradVar), 1e-10 * tf.ones_like(gradVar), gradVar), val)
        for gradVar, val in gradVars
    ]
    accumOps = [
        accumVars[i].assign_add(
            gradVar[0]
        ) if gradVar[0] is not None else accumVars[i]
        for i, gradVar in enumerate(gradVars)
    ]
    trainOp = optimizer.apply_gradients(
        [(accumVars[i], gradVar[1]) for i, gradVar in enumerate(gradVars)]
    )
    return accumOps, zeroOps, trainOp, optimizer


with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        modelX, modelY, modelLoss, prediction = getModel()
        step = 0
        accumOps, zeroOps, trainOp, optimizer = trainOps(modelLoss)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        bar = tqdm(total=1000000)
        prevLoss = 1e10
        for epoch in range(1000):
            batchSize = 64
            totalBatches = int(60000 / batchSize)
            for batch in range(totalBatches):
                x, y = next(dataTrainGen)
                sess.run(zeroOps)
                miniBatchSize = int(batchSize / 4)
                for miniBatch in range(4):
                    xMini = x[miniBatchSize * miniBatch: miniBatchSize * (miniBatch + 1)]
                    yMini = y[miniBatchSize * miniBatch: miniBatchSize * (miniBatch + 1)]
                    sess.run(
                        accumOps,
                        feed_dict={
                            modelX: xMini,
                            modelY: yMini
                        }
                    )
                loss, _ = sess.run(
                    [modelLoss, trainOp],
                    feed_dict={
                        modelX: x,
                        modelY: y
                    }
                )
                bar.update(1)
                bar.set_postfix({"loss": loss})
                step += 1
                if step % 100 == 0:
                    xTest, yTest = next(dataTestGen)
                    lossTest, preds = sess.run(
                        [modelLoss, prediction],
                        feed_dict={
                            modelX: xTest,
                            modelY: yTest
                        }
                    )
                    accuracy = 100 * sum(preds == yTest) / len(preds)
                    if lossTest < prevLoss:
                        print(accuracy, lossTest)
                        saver.save(sess, "Models/mnistModel")
                        prevLoss = lossTest

