import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

LEARNING_RATE = 0.001
BATCH_SIZE = 64
NR_EPOCHS = 50


def plot(accuracies):
    import matplotlib.pyplot as plt

    plt.plot(np.arange(len(accuracies)), accuracies, 'g-')
    plt.show()


def get_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    training_set_x, training_set_y = shuffle(x_train, y_train)

    val_split = int(len(training_set_x) * 49 / 50)

    x_train = training_set_x[:val_split]
    y_train = training_set_y[:val_split]
    x_val = training_set_x[val_split:]
    y_val = training_set_y[val_split:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def scale_dataset(ds):
    retval = list()
    for i in range(len(ds)):
        retval.append(ds[i] / 255)
    return tuple(retval)


def create_probabilities_matrix(targets):
    retval = list()

    for target in targets:
        partial = list()
        for i in range(10):
            if target == i:
                partial.append(1)
            else:
                partial.append(0)
        retval.append(partial)
    return np.array(retval)


def get_batches(x, y, batch_size, rs):
    x_new, y_new = shuffle(x, y, random_state=rs)
    retval = list()

    for i in range(int(len(x) / batch_size)):
        retval.append([np.array(x_new[i * batch_size: (i + 1) * batch_size]),
                       np.array(y_new[i * batch_size: (i + 1) * batch_size])])
    return retval


def run_model(x_t, y_t, x_v, y_v, x, y):
    # Placeholders
    X = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])  # ? x 32 x 32 x 3
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    keep_rate = tf.placeholder(dtype=tf.float32)

    """############################################ LAYERS ############################################"""
    # Convolutional layer 1
    w_1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 32], dtype=tf.float32, stddev=0.1))
    b_1 = tf.Variable(tf.zeros(shape=(32,)))
    a_1 = tf.nn.relu(tf.nn.conv2d(X, w_1, strides=[1, 1, 1, 1], padding='SAME') + b_1)  # ? x 32 x 32 x 32

    # Convolutional layer 2
    w_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], dtype=tf.float32, stddev=0.1))
    b_2 = tf.Variable(tf.zeros(shape=(32,)))
    a_2 = tf.nn.relu(tf.nn.conv2d(a_1, w_2, strides=[1, 1, 1, 1], padding='SAME') + b_2)  # ? x 32 x 32 x 32

    # Max pooling layer 1
    pool_1 = tf.nn.max_pool(a_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # ? x 16 x 16 x 32

    # Dropout layer 1
    drop_1 = tf.nn.dropout(pool_1, keep_rate)

    # Convolutional layer 3
    w_3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], dtype=tf.float32, stddev=0.1))
    b_3 = tf.Variable(tf.zeros(64, ))
    a_3 = tf.nn.relu(tf.nn.conv2d(drop_1, w_3, strides=[1, 1, 1, 1], padding='SAME') + b_3)  # ? x 16 x 16 x 64

    # Convolutional layer 4
    w_4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], dtype=tf.float32, stddev=0.1))
    b_4 = tf.Variable(tf.zeros(64, ))
    a_4 = tf.nn.relu(tf.nn.conv2d(a_3, w_4, strides=[1, 1, 1, 1], padding='SAME') + b_4)  # ? x 16 x 16 x 64

    # Max pooling layer 2
    pool_2 = tf.nn.max_pool(a_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # ? x 8 x 8 x 64

    # Dropout layer 2
    drop_2 = tf.nn.dropout(pool_2, keep_rate)

    # Fully connected layer
    fc_w = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[8 * 8 * 64, 512], stddev=0.1))
    fc_b = tf.Variable(tf.zeros(dtype=tf.float32, shape=(512,)))
    fc_out = tf.nn.relu(tf.matmul(
        tf.reshape(drop_2, [-1, 8 * 8 * 64]),
        fc_w) + fc_b)

    # Dropout layer 3
    drop_3 = tf.nn.dropout(fc_out, keep_rate)

    # Softmax output layer
    out_w = tf.Variable(tf.truncated_normal(shape=[512, 10], stddev=0.1))
    out_b = tf.Variable(tf.zeros(shape=(10,)))
    output = tf.matmul(drop_3, out_w) + out_b

    """####################################### COST + TRAIN #######################################"""
    # Cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=output))

    # Training step
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_step = optimizer.minimize(cost)

    # Accuracy
    hits = tf.equal(tf.argmax(output, axis=1), tf.argmax(Y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(hits, tf.float32))

    """######################################## TRAINING + TESTING ########################################"""
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    accuracies = list()

    for epoch_id in range(NR_EPOCHS):
        batches = get_batches(x_t, y_t, BATCH_SIZE, epoch_id)

        for batch in batches:
            batch_x = batch[0]
            batch_y = batch[1]
            session.run(train_step, feed_dict={X: batch_x,
                                               Y: batch_y,
                                               keep_rate: 0.5})

        v_cost = session.run(cost, feed_dict={X: x_v, Y: y_v, keep_rate: 1})
        v_acc = session.run(accuracy, feed_dict={X: x_v, Y: y_v, keep_rate: 1})
        print(
            "Epoch " + str(epoch_id + 1) + ": Validation cost " + str(v_cost) + ". Validation accuracy: " + str(v_acc))

        accuracies.append(v_acc)

    test_acc = session.run(accuracy, feed_dict={X: x, Y: y, keep_rate: 1})
    print("Final test accuracy: " + str(test_acc))
    session.close()

    plot(accuracies)


def main():
    x_t, y_t, x_v, y_v, x, y = get_dataset()

    # Scaling datasets
    x_t = scale_dataset(x_t)
    x_v = scale_dataset(x_v)
    x = scale_dataset(x)

    # Creating probabilities matrices
    y_t = create_probabilities_matrix(y_t)
    y_v = create_probabilities_matrix(y_v)
    y = create_probabilities_matrix(y)

    run_model(x_t, y_t, x_v, y_v, x, y)


if __name__ == '__main__':
    main()
