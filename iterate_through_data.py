import tensorflow as tf

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    dataset = tf.data.Dataset.range(100)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    for i in range(100):
        value = sess.run(next_element)
        assert i == value

