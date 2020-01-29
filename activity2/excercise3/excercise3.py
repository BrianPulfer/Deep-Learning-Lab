import tensorflow as tf


def main():
    """Using the momentum-based gradient descent"""
    iterations = 20

    learning_rate = tf.constant(1e-1, dtype=tf.float32)
    momentum_coefficient = tf.constant(0.5, dtype=tf.float32)

    momentum = tf.Variable([0, 0], dtype=tf.float32)

    xy_coeff = tf.constant([1/2, 2], dtype=tf.float32)
    xy = tf.Variable([-1000, -1000], dtype=tf.float32)

    z = tf.reduce_sum(tf.multiply(tf.square(xy), xy_coeff))
    grad = tf.gradients(z, xy)[0]

    momentum_update = tf.assign(momentum,
                                            tf.subtract(
                                                tf.multiply(momentum_coefficient, momentum),
                                                tf.multiply(learning_rate, grad)
                                            ))
    xy_update = tf.assign(xy, tf.add(xy, momentum))

    initializer = tf.global_variables_initializer()

    session = tf.Session()
    session.run(initializer)

    for _ in range(iterations):
        session.run(momentum_update)
        new_xy = session.run(xy_update)
        print(new_xy[0], new_xy[1])

    session.close()


if __name__ == '__main__':
    main()
