"""
Initializable iterators take inputs. These inputs are fed into
the initialization operator as a feed_dict.
"""
import tensorflow as tf
from termcolor import cprint

with tf.Session() as sess:
    max_value = tf.placeholder(tf.int64, shape=[])
    dataset = tf.data.Dataset.range(max_value)
    it = dataset.make_initializable_iterator()
    next_element = it.get_next()

    # Initialize an iterator over a dataset with 10 elements.
    sess.run(it.initializer, feed_dict={max_value: 10})
    for i in range(10):
        value = sess.run(next_element)
        assert i == value, "numbers should be correct"

    # Initialize the same iterator over a dataset with 100 elements.
    sess.run(it.initializer, feed_dict={max_value: 100})
    for i in range(100):
        value = sess.run(next_element)
        assert i == value, "numbers should be correct"

cprint("done!", 'green')
