def numpy_solution():
    import numpy as np

    A = np.array([[2, -1, 0],
                 [-1, 2, -1],
                 [0, -1, 2]])
    b = np.array([1, 2, 3])

    x = np.linalg.inv(A).dot(b)
    return x


def tf_solution():
    import tensorflow as tf

    A = tf.constant([[2, -1, 0],
                     [-1, 2, -1],
                     [0, -1, 2]]
                    , dtype=tf.float32)

    b = tf.constant([[1, 2, 3]], dtype=tf.float32)

    result = tf.matmul(tf.matrix_inverse(A), b, transpose_b=True)

    session = tf.Session()
    output = session.run(result)
    session.close()

    return output


def exercise():
    print('Numpy solution', numpy_solution())
    print('Tensorflow solution', tf_solution())


if __name__ == '__main__':
    exercise()
