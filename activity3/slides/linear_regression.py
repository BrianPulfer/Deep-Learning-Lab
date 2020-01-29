import numpy as np
import tensorflow as tf


def create_dataset(sample_size, n_dimensions, sigma, seed=None):
    """Create linear regression dataset (without bias term)"""
    random_state = np.random.RandomState(seed)

    # True weight vector: np.array([1, 2, ..., n_dimensions])
    w = np.arange(1, n_dimensions + 1)

    # Randomly generating observations
    X = random_state.uniform(-1, 1, (sample_size, n_dimensions))

    # Computing noisy targets
    y = np.dot(X, w) + random_state.normal(0.0, sigma, sample_size)

    return X, y


def main():
    sample_size_train = 100
    sample_size_val = 100
    n_dimensions = 10
    sigma = 0.1
    n_iterations = 20
    learning_rate = 0.5

    # Placeholder for the data matrix, where each observation is a row
    X = tf.placeholder(tf.float32, shape=(None, n_dimensions))
    # Placeholder for the targets
    y = tf.placeholder(tf.float32, shape=(None,))

    # Variable for the model parameters
    w = tf.Variable(tf.zeros((n_dimensions, 1)), trainable=True)

    # Loss function
    prediction = tf.reshape(tf.matmul(X, w), (-1,))
    loss = tf.reduce_mean(tf.square(y - prediction))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)  # Gradient descent update operation

    initializer = tf.global_variables_initializer()
    X_train, y_train = create_dataset(sample_size_train, n_dimensions, sigma)

    session = tf.Session()
    session.run(initializer)

    for t in range(1, n_iterations + 1):
        l, _ = session.run([loss, train], feed_dict={X: X_train, y: y_train})
        print('Iteration {0}. Loss: {1}.'.format(t, l))

    X_val, y_val = create_dataset(sample_size_val, n_dimensions, sigma)
    l = session.run(loss, feed_dict={X: X_val, y: y_val})
    print('Validation loss: {0}.'.format(l))
    print(session.run(w).reshape(-1))
    session.close()


if __name__ == '__main__':
    main()
