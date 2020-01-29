import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils


def batch_iterator(X, y, batch_size):
    X = X.reshape(X.shape[0], 784)/255.
    y = utils.to_categorical(y, num_classes=10)

    data = tf.data.Dataset.from_tensor_slices((X, y))
    data = data.shuffle(buffer_size=X.shape[0])
    data = data.repeat()
    data = data.batch(batch_size=batch_size)

    return data.make_one_shot_iterator().get_next()


def main():
    tf.reset_default_graph()
    tf.set_random_seed(seed=0)

    # Loads and splits MNIST dataset
    train_size = 55000
    batch_size = 64
    (X_trainval, y_trainval), (X_test, y_test) = mnist.load_data()
    X_train, y_train = X_trainval[:train_size], y_trainval[:train_size]
    X_val, y_val = X_trainval[train_size:], y_trainval[train_size:]

    train_iter = batch_iterator(X_train, y_train, batch_size)
    # Note: You may want to use smaller batches on a GPU
    val_iter = batch_iterator(X_val, y_val, X_val.shape[0])
    test_iter = batch_iterator(X_test, y_test, X_val.shape[0])  # Subsampling

    # Training procedure hyperparameters
    learning_rate = 1e-3
    n_epochs = 16
    verbose_freq = 2000

    # Model hyperparameters
    n_neurons_1 = 784  # Number of input neurons (28 x 28 x 1)
    n_neurons_2 = 100  # Number of neurons in the second layer (first hidden)
    n_neurons_3 = 100  # Number of neurons in the third layer (second hidden)
    n_neurons_4 = 10 # Number of output neurons (and classes)

    X = tf.placeholder(tf.float32, [None, n_neurons_1])
    Y = tf.placeholder(tf.float32, [None, n_neurons_4])

    # Model parameters. Important: should not be initialized to zero
    W2 = tf.Variable(tf.truncated_normal([n_neurons_1, n_neurons_2]))
    W3 = tf.Variable(tf.truncated_normal([n_neurons_2, n_neurons_3]))
    W4 = tf.Variable(tf.truncated_normal([n_neurons_3, n_neurons_4]))

    b2 = tf.Variable(tf.zeros(n_neurons_2))
    b3 = tf.Variable(tf.zeros(n_neurons_3))
    b4 = tf.Variable(tf.zeros(n_neurons_4))

    # Model definition
    # The rectified linear activation function is given by a = max(z, 0)
    A2 = tf.nn.relu(tf.matmul(X, W2) + b2)
    A3 = tf.nn.relu(tf.matmul(A2, W3) + b3)
    Z4 = tf.matmul(A3, W4) + b4

    # Loss definition
    # Important: this function expects weighted inputs, not activations
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=Z4)
    loss = tf.reduce_mean(loss)

    hits = tf.equal(tf.argmax(Z4, axis=1), tf.argmax(Y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(hits, tf.float32))

    # Using Adam instead of gradient descent
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # Allows saving model to disc
    saver = tf.train.Saver()

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # Using mini-batches instead of entire dataset
    n_batches = n_epochs * (train_size // batch_size)  # roughly
    for t in range(n_batches):
        X_batch, Y_batch = session.run(train_iter)
        session.run(train, {X: X_batch, Y: Y_batch})

        # Computes validation loss every `verbose_freq` batches
        if verbose_freq > 0 and t % verbose_freq == 0: X_batch, Y_batch = session.run(val_iter)
        l = session.run(loss, {X: X_batch, Y: Y_batch})
        print('Batch: {0}. Validation loss: {1}.'.format(t, l))

    saver.save(session, '/tmp/mnist.ckpt')
    session.close()

    # Loading model from file
    session = tf.Session()
    saver.restore(session, '/tmp/mnist.ckpt')

    # In a proper experiment, test set results are computed only once, and
    # absolutely never considered during the choice of hyperparameters
    X_batch, Y_batch = session.run(test_iter)
    acc = session.run(accuracy, {X: X_batch, Y: Y_batch})
    print('Test accuracy: {0}.'.format(acc))
    session.close()


if __name__ == '__main__':
    main()
