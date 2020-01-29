from copy import copy

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def create_dataset(means, std, sample_size, seed=None):
    random_state = np.random.RandomState(seed)

    X = np.zeros((sample_size, len(means[0])), dtype=np.float32)
    Y = np.zeros((sample_size, len(means)), dtype=np.float32)

    cov = np.eye(len(means[0]))*(std**2)

    for i in range(sample_size):
        c = random_state.randint(len(means))
        X[i] = random_state.multivariate_normal(means[c], cov)
        Y[i, c] = 1.
    return X, Y


def a():
    """Create dataset with given parameters in activty (means, std, sample_size, seed)"""
    return create_dataset(means=[(-1, 1), (1, -1)], std=0.5, sample_size=500, seed=0)


def b(X, Y):
    """Plot dataset"""
    x10, y10 = X[Y == [1, 0]].reshape(-1, 2), Y[Y == [1, 0]].reshape(-1, 2)
    x01, y01 = X[Y == [0, 1]].reshape(-1, 2), Y[Y == [0, 1]].reshape(-1, 2)

    plt.scatter(x10[:, 0], x10[:, 1], cmap=plt.cm.RdBu)
    plt.scatter(x01[:, 0], x01[:, 1], cmap=plt.cm.RdBu)

    plt.show()


def c(x, y):
    """Train multilayer perceptron on dataset"""
    # Point C

    TRAINING_STEPS = 100
    LEARNING_RATE = 1e-1

    n_neurons_1 = len(x[0])
    n_neurons_2 = 8
    n_neurons_3 = len(y[0])

    X = tf.placeholder(tf.float32, [None, n_neurons_1])
    Y = tf.placeholder(tf.float32, [None, n_neurons_3])

    w_12 = tf.Variable(tf.truncated_normal(shape=(n_neurons_1, n_neurons_2)))
    b_12 = tf.Variable(tf.zeros(n_neurons_2))

    w_23 = tf.Variable(tf.truncated_normal(shape=(n_neurons_2, n_neurons_3)))
    b_23 = tf.Variable(tf.zeros(n_neurons_3))

    hlo = tf.nn.relu(tf.matmul(X, w_12) + b_12)  # HLO = Hidden Layer's Output
    output = tf.matmul(hlo, w_23) + b_23

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=output))

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    training_step = optimizer.minimize(loss)

    hits = tf.equal(tf.argmax(output, axis=1), tf.argmax(Y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(hits, tf.float32))

    # Session
    saver = tf.train.Saver()

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    for i in range(TRAINING_STEPS):
        acc = session.run(accuracy, feed_dict={X: x, Y: y})
        print('Accuracy at step '+str(i)+': '+str(acc))
        session.run(training_step, feed_dict={X: x, Y: y})

    saver.save(session, 'SavedSession/session.ckpt')
    session.close()

    # Point D
    session = tf.Session()
    tf.train.Saver().restore(session, 'SavedSession/session.ckpt')

    grid_x, grid_y = np.linspace(-3, 3, 100), np.linspace(-3, 3, 100)
    xx, yy = np.meshgrid(grid_x, grid_y)

    observations = np.c_[xx.ravel(), yy.ravel()]

    z = session.run(output, {X: observations})[:, 1]

    for i in range(len(z)):
        if z[i] < 0.5:
            z[i] = 0
        else:
            z[i] = 1

    grid_z = z.reshape(xx.shape)

    plt.contourf(grid_x, grid_y, grid_z)
    #plt.show()


def e():
    pass


def main():
    X, Y = a()
    # b(X, Y)
    c(X, Y)
    b(X, Y)


if __name__ == '__main__':
    main()
    pass
