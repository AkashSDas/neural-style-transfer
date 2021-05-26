import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

# Loading the Tensorflow Hub model for Fast Style Transfer
hub_model = hub.load(
    'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
)


def load_img(path_to_img):
    # Instead of max_dim, frame's height and width can be passed for the resizing
    # step in this function, also 640px is height, width dims for the `fox.mp4` video
    # That's why I've kept max_dim as 640
    max_dim = 640

    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # removing the channel dim and getting other dims from the img shape
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)

    # getting the biggest dim in img shape
    long_dim = max(shape)

    # creating a factor to scale down the long_dim to less than equal to max_dim
    scale = max_dim / long_dim

    # scaling down all dims in the shape tensor, specailly making the long_dim
    # to size equal to max_dim
    new_shape = tf.cast(shape * scale, tf.int32)

    # resizing the img
    img = tf.image.resize(img, new_shape)

    # adding a batch size (240, 240, 3) => (1, 240, 240, 3)
    img = img[tf.newaxis, :]
    return img


def process_tfhub_output(tensor):
    # scaling the img values between 0-255 as the imgs values are
    # currently between 0-1
    tensor = tensor * 255

    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor) > 3:
        # checking for batch size to be 1 (individual img)
        assert tensor.shape[0] == 1

        # getting the img from the batch of size 1
        tensor = tensor[0]
    return tensor


capture = cv2.VideoCapture(0)
style_img = load_img('./style/the_scream.jpg')

while True:
    _, frame = capture.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imsave(f'frame.png', frame, format='png')
    content_img = load_img(f'frame.png')

    # Applying neural style transfer on frame
    result = hub_model(tf.constant(content_img), tf.constant(style_img))
    styled_frame = result[0]

    plt.imsave('frame.png', process_tfhub_output(styled_frame), format='png')

    mat = cv2.imread(f'frame.png', cv2.IMREAD_COLOR)
    umat = cv2.UMat(mat)

    cv2.imshow('Frame', umat)

    # Checking if in 1 millisec the user pressed 'q', if yes then quit
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
capture.destroyAllWindows()
