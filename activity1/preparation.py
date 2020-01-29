import tensorflow as tf


def main():
    # Including constants in the default graph (nodes)
    a = tf.constant([2, 3, 5], dtype=tf.float32)
    b = tf.constant([1, 1, 3], dtype=tf.float32)
    c = tf.constant([1, 2, 2], dtype=tf.float32)

    # Including operations in the default graph (nodes)
    b_plus_c = tf.add(b, c)
    result = tf.multiply(a, b_plus_c)

    # Using operator overloading, we could accomplish the same by writing
    # result = a * (b + c)

    # Creating a TensorFlow session
    session = tf.Session()

    # Using the session to obtain the output for node `result`
    output = session.run(result)    # np.array([4., 9., 25.])

    print(output)

    session.close()


if __name__ == "__main__":
    main()
