"""
Initializable iterators take inputs. These inputs are fed into
the initialization operator as a feed_dict.
"""
import tensorflow as tf
from termcolor import cprint
from ml_logger import logger

with tf.Session() as sess:
    # Define training and validation datasets with the same structure.
    training_dataset = tf.data.Dataset.range(100).map(
        lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
    validation_dataset = tf.data.Dataset.range(50)

    # A reinitializable iterator is defined by its structure. We could use the
    # `output_types` and `output_shapes` properties of either `training_dataset`
    # or `validation_dataset` here, because they are compatible.
    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                               training_dataset.output_shapes)
    next_element = iterator.get_next()

    training_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)

    # Run 20 epochs in which the training dataset is traversed, followed by the
    # validation dataset.
    for _ in range(20):
        # Initialize an iterator over the training dataset.
        sess.run(training_init_op)
        for _ in range(100):
            sess.run(next_element)

        # Initialize an iterator over the validation dataset.
        # logger.split()
        sess.run(validation_init_op)
        # cprint(logger.split(), 'yellow')
        for _ in range(50):
            sess.run(next_element)

    # assert i == value, "numbers should be correct"

cprint("done!", 'green')
