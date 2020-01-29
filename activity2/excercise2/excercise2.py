import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):
    return ((x**2) / 2.) + (2 * y**2)


def create_contour_plot(low=-1000, high=1000, points=50):
    f_range = np.linspace(low, high, points)

    X, Y = np.meshgrid(f_range, f_range)
    Z = f(X, Y)

    plt.contour(X, Y, Z, colors='b')


def main():
    create_contour_plot()

    learning_rate = tf.constant(1e-1, dtype=tf.float32)
    iterations = 20

    coeff = tf.constant([1/2, 2], dtype=tf.float32)
    xy = tf.Variable([[-1000], [-1000]], dtype=tf.float32)
    z = tf.reduce_sum(tf.multiply(tf.square(xy), coeff))

    gradient = tf.gradients(z, xy)[0]
    update = tf.assign(xy, xy - learning_rate * gradient)

    initializer = tf.global_variables_initializer()

    session = tf.Session()
    session.run(initializer)

    for i in range(iterations):
        new_xy = session.run(update)
        x, y = new_xy[0], new_xy[1]
        print(x, y)
        plt.plot(x, y, 'r.-')
    session.close()

    plt.show()


if __name__ == '__main__':
    main()
