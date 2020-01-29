import random
from collections import deque

import gym
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from activity8.atari_wrappers import wrap_atari_deepmind


def copy_parameters_from_online_to_target(session: tf.Session):
    with tf.variable_scope(tf.contrib.framework.get_name_scope(), reuse=True):
        session.run(tf.assign(tf.get_variable('w1_target'), tf.get_variable('w1_online')))
        session.run(tf.assign(tf.get_variable('b1_target'), tf.get_variable('b1_online')))

        session.run(tf.assign(tf.get_variable('w2_target'), tf.get_variable('w2_online')))
        session.run(tf.assign(tf.get_variable('b2_target'), tf.get_variable('b2_online')))

        session.run(tf.assign(tf.get_variable('w3_target'), tf.get_variable('w3_online')))
        session.run(tf.assign(tf.get_variable('b3_target'), tf.get_variable('b3_online')))

        session.run(tf.assign(tf.get_variable('w4_target'), tf.get_variable('w4_online')))
        session.run(tf.assign(tf.get_variable('b4_target'), tf.get_variable('b4_online')))

        session.run(tf.assign(tf.get_variable('w5_target'), tf.get_variable('w5_online')))
        session.run(tf.assign(tf.get_variable('b5_target'), tf.get_variable('b5_online')))


def get_batch(replay_buffer, BATCH_SIZE):
    batch = []
    for i in range(BATCH_SIZE):
        batch.append(replay_buffer[ int(random.random() * len(replay_buffer)) ])
    return batch


def bool_to_int(boolean):
    if boolean:
        return 1
    return 0


def plot_returns_per_episode(returns_per_episode):
    x, y = np.array(returns_per_episode)[:, 0], np.array(returns_per_episode)[:, 1]
    plt.plot(x, y, 'b-')
    plt.show()


def plot_temporal_differences(temporal_difference_errors):
    x = []
    for i in range(len(temporal_difference_errors)):
        x.append(i)

    plt.plot(x, temporal_difference_errors, 'r-')
    plt.show()


def plot_evaluations_scores(evaluations_scores):
    x = []
    for i in range(len(evaluations_scores)):
        x.append(i)

    plt.plot(x, evaluations_scores, 'g-')
    plt.show()


def main():
    """                         Hyperparamteres definition                      """
    REPLAY_BUFFER_SIZE = 10_000
    GAMMA = 0.99
    STEPS = 2_000_000
    EXPLORATION_RATE = 1
    FINAL_EXPLORATION_RATE = 0.1
    SAMPLE_RATE = 4
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    DECAY = 0.99
    COPY_RATE = 10_000 #50_000

    EVALUATION_RATE = 100_000
    EVALUATION_GREEDYNESS = 0.001
    EVALUATION_EPISODES = 5
    EVALUATION_PLAYS = 30

    DEDUCTION = (EXPLORATION_RATE - FINAL_EXPLORATION_RATE) / 1_000_000

    VIDEO_PATH = './video'

    """                         Plotting data structures                      """
    returns_per_episode = []
    temporal_difference_errors = []
    evaluations_scores = []

    """                         Environment creation                      """
    # TODO: Environment should be clipped only during training
    env = wrap_atari_deepmind('BreakoutNoFrameskip-v4', True)
    env.reset()

    # Number of possible actions for BreakOut
    NR_POSSIBLE_ACTIONS = env.action_space.n

    """                         Replay Buffer Initialization                      """
    replay_buffer = deque([], REPLAY_BUFFER_SIZE)

    """                         Network parameters initialization                      """
    X = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32)
    Y = tf.placeholder(shape=[None, ], dtype=tf.float32)
    ACTIONS = tf.placeholder(shape=[None, 2], dtype=tf.int32)

    """ Online Q-Network """
    # Weigths and Biases initialization
    w_initializr, b_initializr = tf.variance_scaling_initializer(), tf.zeros_initializer()


    # Convolutional layer 1 (84x84x4 -> 20x20x32)
    w1 = tf.get_variable(name='w1_online', dtype=tf.float32, initializer=w_initializr(shape=[8, 8, 4, 32]))
    b1 = tf.get_variable(name='b1_online', dtype=tf.float32, initializer=b_initializr(shape=(32,)))
    a1 = tf.nn.relu(tf.nn.conv2d(X, w1, strides=[1, 4, 4, 1], padding='SAME') + b1)

    # Convolutional layer 2 (20x20x32 -> 9x9x64)
    w2 = tf.get_variable(name='w2_online', dtype=tf.float32, initializer=w_initializr(shape=[4, 4, 32, 64]))
    b2 = tf.get_variable(name='b2_online', dtype=tf.float32, initializer=b_initializr(shape=(64,)))
    a2 = tf.nn.relu(tf.nn.conv2d(a1, w2, strides=[1, 2, 2, 1], padding='SAME') + b2)

    # Convolutional layer 3 (9x9x64 -> 7x7x64)
    w3 = tf.get_variable(name='w3_online', dtype=tf.float32, initializer=w_initializr(shape=[3, 3, 64, 64]))
    b3 = tf.get_variable(name='b3_online', dtype=tf.float32, initializer=b_initializr(shape=(64,)))
    a3 = tf.nn.relu(tf.nn.conv2d(a2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)

    # Fully connected layer 1
    w4 = tf.get_variable(name='w4_online', dtype=tf.float32, initializer=w_initializr(shape=[11 * 11 * 64, 512]))
    b4 = tf.get_variable(name='b4_online', dtype=tf.float32, initializer=b_initializr(shape=(512,)))
    a4 = tf.nn.relu(tf.matmul(
        tf.reshape(a3, shape=[-1, 11 * 11 * 64]),
        w4
    ) + b4)

    # Fully connected layer 2
    w5 = tf.get_variable(name='w5_online', dtype=tf.float32, initializer=w_initializr(shape=[512, NR_POSSIBLE_ACTIONS]))
    b5 = tf.get_variable(name='b5_online', dtype=tf.float32, initializer=b_initializr(shape=(NR_POSSIBLE_ACTIONS,)))

    online_q_network_prediction = tf.matmul(a4, w5) + b5

    """ Target Q-Network"""
    # Convolutional layer 1 (84x84x4 -> 20x20x32)
    w1 = tf.get_variable(name='w1_target', dtype=tf.float32, initializer=w_initializr(shape=[8, 8, 4, 32]))
    b1 = tf.get_variable(name='b1_target', dtype=tf.float32, initializer=b_initializr(shape=(32,)))
    a1 = tf.nn.relu(tf.nn.conv2d(X, w1, strides=[1, 4, 4, 1], padding='SAME') + b1)

    # Convolutional layer 2 (20x20x32 -> 9x9x64)
    w2 = tf.get_variable(name='w2_target', dtype=tf.float32, initializer=w_initializr(shape=[4, 4, 32, 64]))
    b2 = tf.get_variable(name='b2_target', dtype=tf.float32, initializer=b_initializr(shape=(64,)))
    a2 = tf.nn.relu(tf.nn.conv2d(a1, w2, strides=[1, 2, 2, 1], padding='SAME') + b2)

    # Convolutional layer 3 (9x9x64 -> 7x7x64)
    w3 = tf.get_variable(name='w3_target', dtype=tf.float32, initializer=w_initializr(shape=[3, 3, 64, 64]))
    b3 = tf.get_variable(name='b3_target', dtype=tf.float32, initializer=b_initializr(shape=(64,)))
    a3 = tf.nn.relu(tf.nn.conv2d(a2, w3, strides=[1, 1, 1, 1], padding='SAME') + b3)

    # Fully connected layer 1
    w4 = tf.get_variable(name='w4_target', dtype=tf.float32, initializer=w_initializr(shape=[11 * 11 * 64, 512]))
    b4 = tf.get_variable(name='b4_target', dtype=tf.float32, initializer=b_initializr(shape=(512,)))
    a4 = tf.nn.relu(tf.matmul(
        tf.reshape(a3, shape=[-1, 11 * 11 * 64]),
        w4
    ) + b4)

    # Fully connected layer 2
    w5 = tf.get_variable(name='w5_target', dtype=tf.float32, initializer=w_initializr(shape=[512, NR_POSSIBLE_ACTIONS]))
    b5 = tf.get_variable(name='b5_target', dtype=tf.float32, initializer=b_initializr(shape=(NR_POSSIBLE_ACTIONS,)))

    target_q_network_prediction = tf.matmul(a4, w5) + b5

    optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay=DECAY)
    loss = tf.reduce_sum((Y - tf.gather_nd(online_q_network_prediction, ACTIONS)) ** 2)
    train_step = optimizer.minimize(
                            loss
                        )

    """                                      Training                                 """
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    copy_parameters_from_online_to_target(session)

    i = 0
    elapsed_episodes = -1
    last_30_episodes_rewards = deque([], 30)

    while i < STEPS:
        elapsed_episodes = elapsed_episodes + 1
        initial_state = env.reset()

        obtained_reward = 0
        episode_finished = 0
        while episode_finished == 0:
            next_action = None
            if random.random() < 1 - EXPLORATION_RATE:
                next_action = np.argmax(session.run(online_q_network_prediction, feed_dict={X: [initial_state]}))
            else:
                next_action = env.action_space.sample()

            next_state, next_reward, episode_finished, _ = env.step(next_action)
            episode_finished = bool_to_int(episode_finished)

            replay_buffer.append([initial_state, next_action, next_reward, next_state, episode_finished])
            i = i + 1

            if EXPLORATION_RATE - DEDUCTION > FINAL_EXPLORATION_RATE:
                EXPLORATION_RATE = EXPLORATION_RATE - DEDUCTION
            else:
                EXPLORATION_RATE = FINAL_EXPLORATION_RATE

            obtained_reward = obtained_reward + next_reward
            initial_state = next_state

            if i >= REPLAY_BUFFER_SIZE:
                if i % SAMPLE_RATE == 0:
                    batch = np.array(get_batch(replay_buffer, BATCH_SIZE))

                    s = batch[:, 0]
                    a = batch[:, 1]
                    r = batch[:, 2]
                    sp = batch[:, 3]
                    fin = batch[:, 4]

                    best_actions = np.max(session.run(target_q_network_prediction, feed_dict={X: list(sp)}), axis=1)

                    y = r - (fin-1) * GAMMA * best_actions
                    #y = tf.Variable(y)

                    taken_actions = []
                    for qq in range(BATCH_SIZE):
                        taken_actions.append([qq, a[qq]])

                    xya_dict = {
                        X: list(s),
                        Y: y,
                        ACTIONS: taken_actions
                    }

                    temporal_difference_errors.append(session.run(loss, feed_dict=xya_dict))
                    session.run(train_step, feed_dict=xya_dict)

                if i % COPY_RATE == 0:
                    copy_parameters_from_online_to_target(session)

                if i % EVALUATION_RATE == 0:
                    # Evaluation
                    eval_env = wrap_atari_deepmind('BreakoutNoFrameskip-v4', False)
                    eval_rewards = []

                    for evaluation_play in range(EVALUATION_PLAYS):
                        play_score = 0
                        for evaluation_ep in range(EVALUATION_EPISODES):
                            eval_state = eval_env.reset()
                            eval_done = 0
                            episode_reward = 0

                            while eval_done == 0:
                                eval_action = None
                                if random.random() < EVALUATION_GREEDYNESS:
                                    eval_action = eval_env.action_space.sample()
                                else:
                                    eval_action = np.argmax(session.run(online_q_network_prediction, feed_dict={X: [eval_state]}))

                                eval_state, eval_r, eval_done, _ = eval_env.step(eval_action)
                                eval_done = bool_to_int(eval_done)

                                episode_reward = episode_reward + eval_r
                            play_score = play_score + episode_reward

                        eval_rewards.append(play_score)

                    score = np.mean(eval_rewards)
                    evaluations_scores.append(score)
                    print("EVALUATION " + str(i / EVALUATION_RATE) + " SCORE: " + str(score))

            if episode_finished == 1:
                last_30_episodes_rewards.append(obtained_reward)
                if len(last_30_episodes_rewards) == 30:
                    moving_average = str(np.mean(last_30_episodes_rewards))
                    returns_per_episode.append([elapsed_episodes, np.mean(last_30_episodes_rewards)])
                    print("Step "+str(i)+", Elapsed episodes = "+str(elapsed_episodes)+", moving average return: "+moving_average+", epsilon: "+str(EXPLORATION_RATE))
                    #last_30_episodes_rewards.clear()
                break

    """                                      Plotting                                 """
    plot_returns_per_episode(returns_per_episode)
    plot_temporal_differences(temporal_difference_errors)
    plot_evaluations_scores(evaluations_scores)

    """                               Rendering of an episode                         """
    monitored_env = gym.wrappers.Monitor(env, VIDEO_PATH, force=True)
    monitored_env.reset()
    state, rew, done, info = monitored_env.step(monitored_env.action_space.sample())

    while not done:
        action = np.argmax(session.run(target_q_network_prediction, feed_dict={X: [state]}))
        state, rew, done, info = monitored_env.step(action)
        #monitored_env.render()

    monitored_env.close()
    """                         Environment destruction                      """
    session.close()
    env.close()


if __name__ == '__main__':
    main()
