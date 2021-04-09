import tensorflow as tf


def getModel():
    x = tf.placeholder(shape=[None, 28, 28], dtype=tf.float32, name="x")
    y = tf.placeholder(shape=[None], dtype=tf.int32, name="y")
    labels = tf.one_hot(y, depth=10)
    xImg = tf.expand_dims(x, axis=-1)
    initializer = tf.random_normal_initializer(stddev=0.1)
    f1 = tf.get_variable("f1", initializer=initializer, shape=[2, 2, 1, 64], dtype=tf.float32)
    conv1 = tf.nn.conv2d(input=xImg, filters=f1, strides=[1, 1, 1, 1], padding="SAME")
    bias = tf.get_variable("bias", initializer=initializer, shape=[64], dtype=tf.float32)
    conv1 = tf.nn.bias_add(conv1, bias)
    conv1 = tf.layers.batch_normalization(conv1)
    conv1 = tf.nn.relu(conv1)
    pool = tf.nn.max_pool2d(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    (_, h, w, f) = pool.get_shape()
    flatten = tf.reshape(pool, [-1, h * w * f])
    denseW1 = tf.get_variable("denseW1", initializer=initializer, shape=[h * w * f, 100], dtype=tf.float32)
    denseB1 = tf.get_variable("denseB1", initializer=initializer, shape=[100], dtype=tf.float32)
    dense1 = tf.nn.xw_plus_b(flatten, denseW1, denseB1)
    denseW2 = tf.get_variable("denseW2", initializer=initializer, shape=[100, 10], dtype=tf.float32)
    denseB2 = tf.get_variable("denseB2", initializer=initializer, shape=[10], dtype=tf.float32)
    dense2 = tf.nn.xw_plus_b(dense1, denseW2, denseB2)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=dense2, labels=labels)
    loss = tf.reduce_mean(loss)
    probs = tf.nn.softmax(dense2)
    prediction = tf.argmax(probs, axis=-1)
    return x, y, loss, prediction

