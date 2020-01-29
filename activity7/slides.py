import numpy as np
import tensorflow as tf


def nback(n, k, length, random_state):
    """Creates n-back task given n, number of digits k, and sequence length.
    Given a sequence of integers `xi`, the sequence `yi` has yi[t] = 1 if and only if xi[t] == xi[t - n].
    """
    xi = random_state.randint(k, size=length) # Input sequence
    yi = np.zeros(length, dtype=int) # Target sequence
    for t in range(n, length):
        yi[t] = (xi[t - n] == xi[t])

    return xi, yi


def nback_dataset(n_sequences, mean_length, std_length, n, k, random_state):
    """Creates dataset composed of n-back tasks."""
    X, Y, lengths = [], [], []

    for _ in range(n_sequences):
        # Choosing length for current task
        length = random_state.normal(loc=mean_length, scale=std_length)
        length = int(max(n + 1, length))

        # Creating task
        xi, yi = nback(n, k, length, random_state)

        # Storing task
        X.append(xi)
        Y.append(yi)
        lengths.append(length)

        # Creating padded arrays for the tasks
        max_len = max(lengths)
        Xarr = np.zeros((n_sequences, max_len), dtype=np.int64)
        Yarr = np.zeros((n_sequences, max_len), dtype=np.int64)

    for i in range(n_sequences):
        Xarr[i, 0: lengths[i]] = X[i]
        Yarr[i, 0: lengths[i]] = Y[i]

    return Xarr, Yarr, lengths


def main():
    seed = 0
    tf.reset_default_graph()
    tf.set_random_seed(seed=seed)

    # Task parameters
    n = 3   # n-back
    k = 4   # Input dimension
    mean_length = 20    # Mean sequence length
    std_length = 5      # Sequence length standard deviation
    n_sequences = 512   # Number of training/validation sequences

    # Creating datasets
    random_state = np.random.RandomState(seed=seed)
    X_train, Y_train, lengths_train = nback_dataset(n_sequences, mean_length,
                                                    std_length, n, k, random_state)

    X_val, Y_val, lengths_val = nback_dataset(n_sequences, mean_length, std_length, n, k, random_state)

    # Model parameters
    hidden_units = 64 # Number of recurrent units
    # Training procedure parameters
    learning_rate = 1e-2
    n_epochs = 256
    # Model definition
    X_int = tf.placeholder(shape=[None, None], dtype=tf.int64)
    Y_int = tf.placeholder(shape=[None, None], dtype=tf.int64)
    lengths = tf.placeholder(shape=[None], dtype=tf.int64)

    batch_size = tf.shape(X_int)[0]
    max_len = tf.shape(X_int)[1]

    # One-hot encoding X_int
    X = tf.one_hot(X_int, depth=k)  # shape: (batch_size, max_len, k)
    # One-hot encoding Y_int
    Y = tf.one_hot(Y_int, depth=2)  # shape: (batch_size, max_len, 2)

    #cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_units)
    cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_units)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # rnn_outputs shape: (batch_size, max_len, hidden_units)
    rnn_outputs, \
    final_state = tf.nn.dynamic_rnn(cell, X, sequence_length=lengths,
                                    initial_state=init_state)

    # rnn_outputs_flat shape: ((batch_size * max_len), hidden_units)
    rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, hidden_units])

    # Weights and biases for the output layer
    Wout = tf.Variable(tf.truncated_normal(shape=(hidden_units, 2), stddev=0.1))
    bout = tf.Variable(tf.zeros(shape=[2]))

    # Z shape: ((batch_size * max_len), 2)
    Z = tf.matmul(rnn_outputs_flat, Wout) + bout

    Y_flat = tf.reshape(Y, [-1, 2])  # shape: ((batch_size * max_len), 2)

    # Creates a mask to disregard padding
    mask = tf.sequence_mask(lengths, dtype=tf.float32)
    mask = tf.reshape(mask, [-1])  # shape: (batch_size * max_len)

    # Network prediction
    pred = tf.argmax(Z, axis=1) * tf.cast(mask, dtype=tf.int64)
    pred = tf.reshape(pred, [-1, max_len])  # shape: (batch_size, max_len)
    hits = tf.reduce_sum(tf.cast(tf.equal(pred, Y_int), tf.float32))
    hits = hits - tf.reduce_sum(1 - mask)  # Disregards padding

    # Accuracy: correct predictions divided by total predictions
    accuracy = hits / tf.reduce_sum(mask)

    # Loss definition (masking to disregard padding)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_flat, logits=Z)
    loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    for e in range(1, n_epochs + 1):
        feed = {X_int: X_train, Y_int: Y_train, lengths: lengths_train}
        l, _ = session.run([loss, train], feed)
        print('Epoch: {0}. Loss: {1}.'.format(e, l))

    feed = {X_int: X_val, Y_int: Y_val, lengths: lengths_val}
    accuracy_ = session.run(accuracy, feed)
    print('Validation accuracy: {0}.'.format(accuracy_))

    # Shows first task and corresponding prediction
    xi = X_val[0, 0: lengths_val[0]]
    yi = Y_val[0, 0: lengths_val[0]]
    print('Sequence:')
    print(xi)
    print('Ground truth:')
    print(yi)
    print('Prediction:')
    print(session.run(pred, {X_int: [xi], lengths: [len(xi)]})[0])
    session.close()


if __name__ == '__main__':
    main()
