import numpy as np
import tensorflow as tf


def generate_batches(text, batch_size, sequence_length):
    # 647 batches, 16 subsequences, 256 subsequence length (chars)
    block_length = len(text) // batch_size
    batches, targets = [], []

    for i in range(0, block_length, sequence_length):
        batch, target = [], []
        for j in range(batch_size):
            start = j * block_length + i
            end = min(start + sequence_length, j * block_length + block_length)

            batch.append(text[start:end])
            target.append(text[start+1: end+1])
        batches.append(np.array(batch, dtype=int))
        targets.append(target)
    return batches, targets


def preprocess_text():
    nr_unique_chars, frequencies, text = 0, dict(), []

    file = open("./TheCountofMonteCristo.txt", 'r')

    for line in file:
        for char in line:
            char = char.lower()
            text.append(char_to_int(char))

            if char in frequencies:
                frequencies[char] = frequencies[char] + 1
            else:
                frequencies[char] = 1

    nr_unique_chars = len(frequencies.keys())
    file.close()
    return nr_unique_chars, frequencies, text


def char_to_int(char):
    return ord(char)   # a = 1, b = 2, ...


def int_to_char(integer):
    return chr(integer)


def integers_to_string(int_array):
    retval = ''
    for nr in int_array:
        retval += int_to_char(nr)
    return retval


def remove_last(batches, targets):
    batches.remove(batches[-1])
    targets.remove(targets[-1])
    return batches, targets


def main():
    tf.reset_default_graph()
    tf.set_random_seed(seed=0)

    # HYPERPARAMETERS
    LEARNING_RATE = 0.01    # 1e-2
    NR_EPOCHS = 1
    NR_UNITS_LSTM = 256

    NR_TEST_SENTENCES = 20
    LENGTH_TEST_SENTENCES = 256

    # TEXT PRE-PROCESSING AND BATCHES CREATION
    nr_batches, subsequences_size = 16, 256
    nr_unique_chars, frequencies, text = preprocess_text()

    batches, targets = generate_batches(text, nr_batches, subsequences_size)
    batches, targets = remove_last(batches, targets)

    # COMPUTATIONAL GRAPH CREATION
    X_int = tf.placeholder(shape=[None, None], dtype=tf.int64)
    Y_int = tf.placeholder(shape=[None, None], dtype=tf.int64)
    lengths_placeholder = tf.placeholder(shape=[None], dtype=tf.int64)

    batch_size = tf.shape(X_int)[0]
    max_len = tf.shape(X_int)[1]

    X = tf.one_hot(X_int, depth=nr_unique_chars)
    Y = tf.one_hot(Y_int, depth=nr_unique_chars)

    lstm1, lstm2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=NR_UNITS_LSTM), tf.nn.rnn_cell.BasicLSTMCell(num_units=NR_UNITS_LSTM)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm1, lstm2])

    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    rnn_outputs, \
    final_state = tf.nn.dynamic_rnn(cell, X, sequence_length=lengths_placeholder,
                                    initial_state=init_state)

    rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, NR_UNITS_LSTM])

    Wout = tf.Variable(tf.truncated_normal(shape=(NR_UNITS_LSTM, nr_unique_chars), stddev=0.1))
    bout = tf.Variable(tf.zeros(shape=[nr_unique_chars]))

    Z = tf.matmul(rnn_outputs_flat, Wout) + bout

    Y_flat = tf.reshape(Y, [-1, nr_unique_chars])

    mask = tf.sequence_mask(lengths_placeholder, dtype=tf.float32)
    mask = tf.reshape(mask, [-1])

    pred = tf.argmax(Z, axis=1) * tf.cast(mask, dtype=tf.int64)
    pred = tf.reshape(pred, [-1, max_len])
    hits = tf.reduce_sum(tf.cast(tf.equal(pred, Y_int), tf.float32))
    hits = hits - tf.reduce_sum(1 - mask)

    accuracy = hits / tf.reduce_sum(mask)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_flat, logits=Z)
    loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train = optimizer.minimize(loss)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # Training
    for epoch in range(NR_EPOCHS):
        current_state = session.run([init_state], feed_dict={X_int: batches[0]})
        for batch in range(len(batches)):
            current_state, loss_value, _ = session.run([final_state, loss, train], feed_dict={
                X_int: batches[batch],
                Y_int: targets[batch],
                init_state: current_state,
                lengths_placeholder: [256] * 16,
            })
            print('Epoch: {0}. Batch: {1}. Loss: {2}.'.format(epoch, batch, loss_value))

    # Testing
    for i in range(NR_TEST_SENTENCES):
        curr_char = [[i+96]]
        sentence = [int_to_char(curr_char[0][0])]
        state = session.run([init_state], feed_dict={X_int: curr_char})
        for j in range(LENGTH_TEST_SENTENCES):
            state, curr_char = session.run([final_state, pred], feed_dict={
                X_int: curr_char,
                lengths_placeholder: [1],
                init_state: state
            })
            sentence += int_to_char(curr_char[0][0])

        print(sentence)
    session.close()


if __name__ == '__main__':
    main()
