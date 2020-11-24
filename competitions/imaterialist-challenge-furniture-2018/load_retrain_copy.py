from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
import json
from random import randint

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from settings import BASE_DIR, DATA_DIR, BASE_DIR_KAGGLE
from appsettings import COMPETITION_NAME,\
    ROOT_DATA_DIR, COMPETITION_DATA,\
    COMPETITION_FILE_LIST, TEST_DATA,\
    TRAIN_DATA, VAL_DATA, IMAGE_PATH,\
    LABEL_FILE, PROCESSED_IMAGE_PATH
from image.resize import resize_np2np
from image.rgbconversion import rgb_path2nparray, rgb_path2path


COMP_DIR = os.path.join(BASE_DIR_KAGGLE, "competitions", COMPETITION_NAME)
COMP_DATA_DIR = os.path.join(DATA_DIR, ROOT_DATA_DIR, COMPETITION_DATA)
TRAIN_IMAGE_PATH = os.path.join(DATA_DIR, ROOT_DATA_DIR, TRAIN_DATA, IMAGE_PATH)
TEST_IMAGE_PATH = os.path.join(DATA_DIR, ROOT_DATA_DIR, TEST_DATA, IMAGE_PATH)
VAL_IMAGE_PATH = os.path.join(DATA_DIR, ROOT_DATA_DIR, VAL_DATA, IMAGE_PATH)
TRAIN_LABEL_PATH = os.path.join(DATA_DIR, ROOT_DATA_DIR, TRAIN_DATA, LABEL_FILE)
TEST_LABEL_PATH = os.path.join(DATA_DIR, ROOT_DATA_DIR, TEST_DATA, LABEL_FILE)
VAL_LABEL_PATH = os.path.join(DATA_DIR, ROOT_DATA_DIR, VAL_DATA, LABEL_FILE)

IMAGE_X = 64
IMAGE_Y = 64
IMAGE_LAYER = 3
CLASS_MIN = 1
CLASS_MAX = 256
TRAIN_LABEL_DF = None
VAL_LABEL_DF = None
TRAIN_SAMPLES = 0
VAL_SAMPLES = 0
TEST_SAMPLES = 0
SUBMISSION_FP = None
SUBMISSION_FPATH = ""

MAX_SAMPLES = 100
MAX_SAMPLES_TRAIN = 100
MAX_SAMPLES_VAL = 10
MAX_SAMPLES_TEST = 10

NUM_EPOCHS = 1
MINI_BATCH_SIZE = 10


TF_MODEL_DIR = os.path.join(COMP_DIR, "tfhub_inception_model")
TF_PRED_OUT_DIR="kaggle_submission"
TF_HUB_MODULE_URL = "https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1"
tf_module_name = "inception_resnet_v2"
TF_MODULE = None
# The location where variable checkpoints will be stored.
TF_CHECKPOINT_NAME = os.path.join(TF_MODEL_DIR, "_retrain_checkpoint")
# A module is understood as instrumented for quantization with TF-Lite
# if it contains any of these ops.
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')

tf_final_tensor_name = "furniture_classification"
tf_learning_rate = 0.001
tf_bottleneck_dir = os.path.join(TF_MODEL_DIR, "tfhub_bottleneck_dir")
tf_bottleneck_dir_train = os.path.join(tf_bottleneck_dir, "train")
tf_bottleneck_dir_test = os.path.join(tf_bottleneck_dir, "test")
tf_bottleneck_dir_validation = os.path.join(tf_bottleneck_dir, "validation")
tf_summaries_dir = os.path.join(TF_MODEL_DIR, "tensorflow_summaries")
tf_training_steps = 200
tf_eval_step_interval = 50
tf_intermediate_store_frequency = 50
tf_intermediate_output_graphs_dir = os.path.join(TF_MODEL_DIR, "intermediate_output")
tf_class_count = 256
tf_output_graph = os.path.join(TF_MODEL_DIR, "output_graph")
tf_saved_model_dir = os.path.join(TF_MODEL_DIR, "saved_model1")

TF_SAVED_MODEL_DIR = os.path.join(DATA_DIR, ROOT_DATA_DIR, "models", "saved_model3")


def verifyImage(path):
    if not os.path.isfile(path):
        return False
    rgb_img = rgb_path2nparray(path)
    if rgb_img is None:
        return False
    try:
        img_rsz = resize_np2np(rgb_img, (IMAGE_X, IMAGE_Y))
        img_shape = img_rsz.shape
        if len(img_shape) == 3 and img_shape[2] == 3:  # (RGB images)
            return True
        else:
            return False
    except:
        return False


def getDatasetArrays(image_filepath, max_samples=1000):
    fnames = []
    image_ids = []
    for i in range(1, max_samples):
        image_location = os.path.join(image_filepath, str(i))
        processed_location = image_location + ".jpeg"
        if not os.path.exists(processed_location):
            if verifyImage(image_location):
                if rgb_path2path(image_location, processed_location, overwrite=False) == 0:
                    image_ids.append(i)
                    fnames.append(os.path.join(processed_location))
        else:
            image_ids.append(i)
            fnames.append(os.path.join(processed_location))
    return image_ids, fnames


def add_jpeg_decoding(module_spec):
    """Adds operations that perform JPEG decoding and resizing to the graph..

    Args:
    module_spec: The hub.ModuleSpec for the image module being used.

    Returns:
    Tensors for the node to feed JPEG data into, and the output of the
      preprocessing steps.
    """
    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    # Convert from full range of uint8 to range [0,1] of float32.
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                          tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    return jpeg_data, resized_image


if __name__ == '__main__':

    with open(os.path.join(COMP_DATA_DIR, 'test.json')) as fp:
        test_json = json.loads(fp.read())
        TEST_SAMPLES = len(test_json['images'])

    module_spec = hub.load_module_spec(TF_HUB_MODULE_URL)
    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)

    test_image_ids, test_filenames = getDatasetArrays(TEST_IMAGE_PATH, max_samples=10)

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], TF_SAVED_MODEL_DIR)
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(module_spec)
        graph = tf.get_default_graph()
        model = graph.get_tensor_by_name("furniture_classification:0")
        image_path = test_filenames[0]
        if not tf.gfile.Exists(image_path):
            tf.logging.fatal('Processed File does not exist %s', image_path)
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        resized_input_values = sess.run(decoded_image_tensor,
                                        {jpeg_data_tensor: image_data})
        resized_input_tensor = tf.placeholder(tf.float32, [None, input_height, input_width, input_depth])

        print(tf.shape(resized_input_values))

        """
        print(sess.run(model, {
            resized_input_tensor: resized_input_values
        }))

        """
