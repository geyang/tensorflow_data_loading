import tensorflow as tf
from PIL import Image
from termcolor import cprint

string_tensor = tf.convert_to_tensor(['../figures/PointMass-v0.png'])
filename_queue = tf.data.Dataset.from_tensor_slices(string_tensor).shuffle(string_tensor.shape[0]).repeat(10)


# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized, label


# A vector of filenames.
filenames = tf.constant(['test-dataset/PointMass-v0.png'])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([37])

iterator = tf.data.Dataset \
    .from_tensor_slices((filenames, labels)) \
    .map(_parse_function) \
    .make_one_shot_iterator()

next_element = iterator.get_next()

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    for i in range(1):
        img, lable = sess.run(next_element)

        im = Image.fromarray(img.reshape(img.shape[:2]))
        im.show()
