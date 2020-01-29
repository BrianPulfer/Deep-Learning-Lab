import numpy as np
import tensorflow as tf
import os


def plot_training_set_and_estimations(x_train, y_train, w_star, w_estimation):
    import matplotlib.pyplot as plt

    xs = x_train[:, 1]

    y_star = np.dot(x_train, w_star)
    y_estimate = np.dot(x_train, w_estimation)

    lower_bound = np.min([np.min(y_train), np.min(y_star), np.min(y_estimate)]) - 1
    upper_bound = np.max([np.max(y_train), np.max(y_star), np.max(y_estimate)]) + 1
    boud_y_axis = [-3, 2, lower_bound, upper_bound]

    plt.plot(xs, y_train, 'ro')
    plt.axis(boud_y_axis)
    plt.show()
    plt.close()

    plt.plot(xs, y_train, 'ro', xs, y_star, 'go')
    plt.axis(boud_y_axis)
    plt.show()
    plt.close()

    plt.plot(xs, y_train, 'ro', xs, y_estimate, 'bo')
    plt.axis(boud_y_axis)
    plt.show()
    plt.close()

    plt.plot(xs, y_train, 'ro', xs, y_star, 'go', xs, y_estimate, 'bo')
    plt.axis(boud_y_axis)
    plt.show()
    plt.close()


def create_dataset(w_star, x_range, sample_size, sigma, seed=None):
    random_state = np.random.RandomState(seed)

    x = random_state.uniform(x_range[0], x_range[1], sample_size)
    X = np.zeros((sample_size, w_star.shape[0]))

    for i in range(sample_size):
        X[i, 0] = 1.
        for j in range(1, w_star.shape[0]):
            X[i, j] = x[i] ** j

    y = X.dot(w_star)
    if sigma > 0:
        y += random_state.normal(0.0, sigma, sample_size)

    return X, y


if __name__ == '__main__':
    steps = 1000    # 800 | 1100
    learning_rate = 0.011   # 0.011 | 0.010
    nr_observations = 200
    sigma = 1/2

    w_star = np.array([-8, -4, 2, 1])
    x_range = np.array([-3, 2])

    x_train, y_train = create_dataset(w_star, x_range, nr_observations, sigma, seed=0)
    x_val, y_val = create_dataset(w_star, x_range, nr_observations, sigma, seed=1)

    x_train_bonus, y_train_bonus = create_dataset(np.append(w_star, [0]), x_range, nr_observations, sigma, seed=0)
    x_val_bonus, y_val_bonus = create_dataset(np.append(w_star, [0]), x_range, nr_observations, sigma, seed=1)

    features_nr = x_train.shape[1]

    X = tf.placeholder(tf.float32, shape=(None, features_nr))
    y = tf.placeholder(tf.float32, shape=(None,))

    X_bonus = tf.placeholder(tf.float32, shape=(None, features_nr+1))
    y_bonus = tf.placeholder(tf.float32, shape=(None, ))

    w = tf.Variable(tf.zeros(shape=(features_nr, 1)))
    w_bonus = tf.Variable(tf.zeros(shape=(features_nr+1, 1)))

    prediction = tf.reshape(tf.matmul(X, w), (-1,))
    loss = tf.reduce_mean(tf.square(y - prediction))

    prediction_bonus = tf.reshape(tf.matmul(X_bonus, w_bonus), (-1,))
    loss_bonus = tf.reduce_mean(tf.square(y_bonus - prediction_bonus), (-1,))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(loss)

    optimizer_bonus = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_step_bonus = optimizer_bonus.minimize(loss_bonus)

    initializer = tf.global_variables_initializer()

    #directory = './tmp/gradient_descent'
    #os.makedirs(directory)

    t_loss = tf.Variable(0, dtype=tf.float32)
    v_loss = tf.Variable(0, dtype=tf.float32)

    t_loss_bonus = tf.Variable(0, dtype=tf.float32)
    v_loss_bonus = tf.Variable(0, dtype=tf.float32)

    #tf.summary.scalar('training loss', t_loss)
    #tf.summary.scalar('validation loss', v_loss)

    #summaries = tf.summary.merge_all()

    session = tf.Session()

    #writer = tf.summary.FileWriter(directory, session.graph)
    session.run(initializer)

    for step in range(steps):
        loss_val = session.run(loss, feed_dict={X: x_val, y: y_val})
        loss_train = session.run(loss, feed_dict={X: x_train, y: y_train})

        loss_val_bonus = session.run(loss_bonus, feed_dict={X_bonus: x_val_bonus, y_bonus: y_val_bonus})
        loss_train_bonus = session.run(loss_bonus, feed_dict={X_bonus: x_train_bonus, y_bonus: y_train_bonus})

        if step == 0:
            print("Initial Loss in training set: ", format(loss_train))
            print("Initial Loss in validation set: ", format(loss_val))
            print("Initial Loss in train Bonus set: ", format(loss_train_bonus))
            print("Initial Loss in validation Bonus set: ", format(loss_val_bonus))

        session.run(train_step, feed_dict={X: x_train, y: y_train})
        session.run(train_step_bonus, feed_dict={X_bonus: x_train_bonus, y_bonus: y_train_bonus})

        session.run(tf.assign(t_loss, loss_train))
        session.run(tf.assign(v_loss, loss_val))
        #s = session.run(summaries)
        #writer.add_summary(s, step)

   # writer.close()

    loss_train = session.run(loss, feed_dict={X: x_train, y: y_train})
    loss_val = session.run(loss, feed_dict={X: x_val, y: y_val})
    print("Final Loss in training set: ", format(loss_train))
    print("Final Loss in validation set: ", format(loss_val))

    loss_train_bonus = session.run(loss_bonus, feed_dict={X_bonus: x_train_bonus, y_bonus: y_train_bonus})
    loss_val_bonus = session.run(loss_bonus, feed_dict={X_bonus: x_val_bonus, y_bonus: y_val_bonus})
    print("Final Loss in training set BONUS: ", format(loss_train_bonus))
    print("Final Loss in validation set BONUS: ", format(loss_val_bonus))

    final_weights = session.run(w).reshape(-1)
    # print("Final weights: ")
    # print(final_weights)

    final_weights_bonus = session.run(w_bonus).reshape(-1)
    session.close()

    plot_training_set_and_estimations(x_val, y_val, w_star.reshape((4, 1)), final_weights.reshape((4, 1)))
    plot_training_set_and_estimations(x_val_bonus,
                                      y_val_bonus,
                                      np.append(w_star, [0]).reshape((5, 1)),
                                      final_weights_bonus.reshape(5, 1))
