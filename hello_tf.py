import tensorflow as tf


# 打印
def printf_info():
    print("Using TensorFlow version %s" % tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# hello world
def hello_tf_v1():
    # 兼容v1版本，否则run会报错
    tf.compat.v1.disable_eager_execution()

    message = tf.constant('Hello, TensorFlow!')

    sess = tf.compat.v1.Session()

    print(sess.run(message))


# 记录运算运行在哪一个设备上
def log_device_placement():
    tf.debugging.set_log_device_placement(True)

    # Create some tensors
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

    print(c)


# 将tf运行在指定cpu上
def place_tensors_on_cpu():
    tf.debugging.set_log_device_placement(True)

    # Place tensors on the CPU
    with tf.device('/CPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    c = tf.matmul(a, b)
    print(c)


def main():
    printf_info()
    hello_tf_v1()
    # log_device_placement()
    # place_tensors_on_cpu()


if __name__ == "__main__":
    main()
