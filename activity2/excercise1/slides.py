import os

import tensorflow as tf


def slide21():
    """Slide 21 - Variables"""
    a = tf.Variable([1.0, 1.0, 1.0], dtype=tf.float32)
    b = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)

    c = a * b

    # Operation that assigns initial values to all variables (a)
    initialize = tf.global_variables_initializer()

    # Operations that assigns 2*a to a
    double_a = tf.assign(a, 2*a)

    session = tf.Session()

    # Obtains initialize output. Side-effect: initializes a
    session.run(initialize)
    print(session.run(c))   # [1,2,3]

    # Obtains 'double_a' output. Side-effect: doubles a
    session.run(double_a)
    print(session.run(c))   # [2,4,6]
    session.run(double_a)
    print(session.run(c))   # [4,8,12]

    session.close()

    session = tf.Session()
    session.run(initialize)
    print(session.run(c))   # np.array([1.0, 2.0, 3.0])
    session.close()


def slide22():
    """Slide 22 - Placeholders"""
    a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    b = tf.placeholder(dtype=tf.float32)  # Placeholder, shape omitted

    c = a * b

    session = tf.Session()
    print(session.run(c, feed_dict={b: 2.0}))  # np.array([2.0, 4.0, 6.0])
    print(session.run(c, feed_dict={b: [1.0, 2.0, 3.0]}))  # np.array([1.0, 4.0, 9.0])
    session.close()


def slide23():
    """Slide 23 - Gradients"""
    x = tf.Variable([1.0, 2.0, 3.0])
    y = tf.reduce_sum(tf.square(x))

    grad = tf.gradients(y, x)[0]  # Gradient of y wrt `x`

    initializer = tf.global_variables_initializer()

    session = tf.Session()
    session.run(initializer)
    print(session.run(grad))  # np.array([2.0, 4.0, 6.0])
    session.close()


def slide25():
    """Slide 25 - Gradient Descent"""
    n_iterations = 20

    learning_rate = tf.constant(1e-1, dtype=tf.float32)  # Learning rate 0.1

    # Goal: finding x such that y is minimum
    x = tf.Variable([0.0, 0.0, 0.0])  # Initial guess
    y = tf.reduce_sum(tf.square(x - tf.constant([1.0, 2.0, 3.0])))

    grad = tf.gradients(y, x)[0]

    update = tf.assign(x, x - learning_rate * grad)  # Gradient descent update

    initializer = tf.global_variables_initializer()
    session = tf.Session()

    session.run(initializer)
    for _ in range(n_iterations):
        session.run(update)
        print(session.run(x))  # State of `x` at this iteration
    session.close()


def slide28_29():
    """Slide 28/29"""
    directory = './tmp/gradient_descent'  # Directory for data storage
    os.makedirs(directory)

    n_iterations = 20

    # Naming constants/variables to facilitate inspection
    learning_rate = tf.constant(1e-1, dtype=tf.float32, name='learning_rate')
    x = tf.Variable([0.0, 0.0, 0.0], name='x')
    target = tf.constant([1.0, 2.0, 3.0], name='target')
    y = tf.reduce_sum(tf.square(x - target))

    grad = tf.gradients(y, x)[0]

    update = tf.assign(x, x - learning_rate * grad)

    tf.summary.scalar('y', y)  # Includes summary attached to `y`
    tf.summary.scalar('x_1', x[0]) # Includes summary attached to `x[0]`
    tf.summary.scalar('x_2', x[1]) # Includes summary attached to `x[1]`
    tf.summary.scalar('x_3', x[2]) # Includes summary attached to `x[2]`

    # Merges all summaries into single a operation
    summaries = tf.summary.merge_all()

    initializer = tf.global_variables_initializer()

    session = tf.Session()

    # Creating object that writes graph structure and summaries to disk
    writer = tf.summary.FileWriter(directory, session.graph)

    session.run(initializer)

    for t in range(n_iterations):
        # Updates `x` and obtains the summaries for iteration t
        s, _ = session.run([summaries, update])

        # Stores the summaries for iteration t
        writer.add_summary(s, t)

    print(session.run(x))
    writer.close()
    session.close()

    # Run tensorboard --logdir="/tmp/gradient_descent" --port 6006
    # Access http://localhost:6006 and see scalars/graphs


if __name__ == '__main__':
    # slide21()
    # slide22()
    # slide23()
    # slide25()
    slide28_29()
    pass
