import tensorflow as tf
from termcolor import cprint

with tf.Session() as sess:
    dataset = tf.data.Dataset.range(100)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    for i in range(100):
        value = sess.run(next_element)
        assert i == value, "numbers should be correct"

cprint("done!", 'green')
