from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import os
import pandas as pd
import tensorflow as tf
from random import randint
import time
import sys
import json

from settings import BASE_DIR, DATA_DIR, BASE_DIR_KAGGLE
from appsettings import COMPETITION_NAME,\
    ROOT_DATA_DIR, COMPETITION_DATA,\
    COMPETITION_FILE_LIST, TEST_DATA,\
    TRAIN_DATA, VAL_DATA, IMAGE_PATH,\
    LABEL_FILE
from image.resize import resize_np2np
from image.rgbconversion import rgb_path2nparray, rgb_path2path

# globals
COMP_DIR=os.path.join(BASE_DIR_KAGGLE, "competitions", COMPETITION_NAME)
COMP_DATA_DIR=os.path.join(DATA_DIR, ROOT_DATA_DIR, COMPETITION_DATA)
TRAIN_IMAGE_PATH=os.path.join(DATA_DIR, ROOT_DATA_DIR, TRAIN_DATA, IMAGE_PATH)
TEST_IMAGE_PATH=os.path.join(DATA_DIR, ROOT_DATA_DIR, TEST_DATA, IMAGE_PATH)
VAL_IMAGE_PATH=os.path.join(DATA_DIR, ROOT_DATA_DIR, VAL_DATA, IMAGE_PATH)
TRAIN_LABEL_PATH=os.path.join(DATA_DIR, ROOT_DATA_DIR, TRAIN_DATA, LABEL_FILE)
TEST_LABEL_PATH=os.path.join(DATA_DIR, ROOT_DATA_DIR, TEST_DATA, LABEL_FILE)
VAL_LABEL_PATH=os.path.join(DATA_DIR, ROOT_DATA_DIR, VAL_DATA, LABEL_FILE)


# GLOBALS
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

TF_MODEL_DIR="./tensorflow_models"
TF_PRED_OUT_DIR="kaggle_submission"


def convulation_output_grid(conv_out, img_size, conv_layers):
    x_img = img_size[0]
    y_img = img_size[1]
    channels = img_size[2]

    grid = tf.slice(conv_out, (0, 0, 0, 0), (1, -1, -1, -1))
    grid = tf.reshape(grid, (img_size, img_size, channels))

    # Reorder so the channels are in the first dimension, x and y follow.
    grid = tf.transpose(grid, (2, 0, 1))
    # Bring into shape expected by image_summary
    grid = tf.reshape(grid, (-1, img_size, img_size, 1))
    return grid


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


def getSubmissionFile():
    tstamp = str(int(time.time()))
    sub_dir = os.path.join(COMP_DIR, TF_PRED_OUT_DIR)
    if not os.path.isdir(sub_dir):
        try:
            os.mkdir(sub_dir)
        except Exception as e:
            print("Can not create submission dir")
            print(e)
    fp_path = os.path.join(COMP_DIR, TF_PRED_OUT_DIR, tstamp + ".csv")
    try:
        fp = open(fp_path, "w")
        return fp, fp_path
    except Exception as e:
        print(e)
        return None, fp_path


def getOneHotVector(hotone, max_class=256):
    retval = [0]*max_class
    retval[hotone] = 1
    return retval


def getRandomSample(selected_sofar, maxdata, image_filepath):
    rejected_so_far = []
    while True:
        if len(selected_sofar) + len(rejected_so_far) == maxdata:
            break
        randsample = randint(1, maxdata - 1)
        if randsample in selected_sofar:
            rejected_so_far.append(randsample)
            continue
        image_location = os.path.join(image_filepath, str(randsample))
        if verifyImage(image_location):
            return randsample
    return selected_sofar[0]


def getDatasetArrays(label_df, image_filepath, total_samples, max_samples=1000):
    fnames = []
    flabels = []
    selected_so_far = []
    image_ids = []
    for _ in range(max_samples):
        randsample = getRandomSample(selected_so_far, total_samples, image_filepath)
        selected_so_far.append(randsample)
        image_ids.append(randsample)
        fnames.append(os.path.join(image_filepath, str(randsample)))
        if label_df is not None:
            # flabels.append(getOneHotVector(int(label_df['predicted'][randsample - 1])))
            flabels.append(int(label_df['predicted'][randsample - 1]))
    return image_ids, fnames, flabels


# Parse funciton for dataset input functio
def _read_py_function(filename, label):
    image_decoded = rgb_path2nparray(filename.decode())
    image_resized = resize_np2np(image_decoded, (IMAGE_X, IMAGE_Y))
    return image_resized, label


def _read_py_test_function(filename, imid):
    image_decoded = rgb_path2nparray(filename.decode())
    image_resized = resize_np2np(image_decoded, (IMAGE_X, IMAGE_Y))
    return image_resized, imid


# Use standard TensorFlow operations to resize the image to a fixed shape.
def _resize_function(image_decoded, label):
    image_decoded.set_shape([IMAGE_X, IMAGE_Y, IMAGE_LAYER])
    return tf.cast(tf.convert_to_tensor(image_decoded), dtype=tf.float32), label


def dataset_input_fn(filenames, flabels):
    filenames = tf.constant(filenames)
    class_labels = tf.constant(flabels)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, class_labels))
    dataset = dataset.map(
        lambda filename, label: tuple(tf.py_func(
            _read_py_function, [filename, label], [tf.uint8, label.dtype])))
    dataset = dataset.map(_resize_function)
    dataset = dataset.shuffle(buffer_size=1000)
    batched_dataset = dataset.batch(MINI_BATCH_SIZE)
    batched_dataset = batched_dataset.repeat(NUM_EPOCHS)
    iterator = batched_dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return {"image_data": features}, labels


def train_input_fn():
    _, filenames, flabels = getDatasetArrays(TRAIN_LABEL_DF, TRAIN_IMAGE_PATH,
                                             TRAIN_SAMPLES, max_samples=MAX_SAMPLES_TRAIN)
    return dataset_input_fn(filenames, flabels)


def val_input_fn():
    _, filenames, flabels = getDatasetArrays(VAL_LABEL_DF, VAL_IMAGE_PATH,
                                             VAL_SAMPLES, max_samples=MAX_SAMPLES_VAL)
    return dataset_input_fn(filenames, flabels)


def test_input_fn():
    image_ids, filenames, _ = getDatasetArrays(None, TEST_IMAGE_PATH,
                                             TEST_SAMPLES, max_samples=MAX_SAMPLES_TEST)
    filenames = tf.constant(filenames)
    image_ids = tf.constant(image_ids)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, image_ids))
    dataset = dataset.map(
        lambda filename, imid: tuple(tf.py_func(
            _read_py_test_function, [filename, imid], [tf.uint8, tf.int32])))
    dataset = dataset.map(_resize_function)
    dataset = dataset.shuffle(buffer_size=1000)
    batched_dataset = dataset.batch(MINI_BATCH_SIZE)
    batched_dataset = batched_dataset.repeat(NUM_EPOCHS)
    iterator = batched_dataset.make_one_shot_iterator()
    filearray, fileid = iterator.get_next()
    return {
        "image_data": filearray,
        "image_id": fileid
    }


def factorize(val):
    return [(i, val // i) for i in range(1, int(val**0.5)+1) if val % i == 0]


def get_conv_output_grid(conv_out, x, y, f):
    # conv_out dim([bacth_size, x, y, f])
    factors = factorize(f)[-1]
    t_arr = tf.unstack(conv_out, axis=3)  # dim([-1, x, y]) array of size f
    grid_cols = []
    start = 0
    end = start + factors[0]
    for i in range(factors[1]):
        for j in range(start, end):
            if len(grid_cols) == i:
                grid_cols.append(t_arr[j])
            else:
                grid_cols[i] = tf.concat([grid_cols[i], t_arr[j]], axis=0)
        start = end
        end = start + factors[0]
    grid = grid_cols[0]
    for i in range(1, factors[1]):
        grid = tf.concat([grid, grid_cols[i]], axis=1)
    grid = tf.reshape(grid, (MINI_BATCH_SIZE, x*factors[0], y*factors[1], 1))
    return grid

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["image_data"], [-1, IMAGE_X, IMAGE_Y, IMAGE_LAYER])

    tf.summary.image("input_layer_image", input_layer, max_outputs=5)

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    grid = get_conv_output_grid(conv1, IMAGE_X, IMAGE_Y, 32)
    tf.summary.image('conv1_weights', grid, max_outputs=5)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    grid2 = get_conv_output_grid(conv2, IMAGE_X//2, IMAGE_Y//2, 64)
    tf.summary.image('conv2_weights', grid2, max_outputs=5)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, IMAGE_X//4 * IMAGE_Y//4 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=256)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "image_id": features["image_id"],
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
    else:
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):

    user_opt = 0
    try:
        if len(unused_argv) > 1:
            user_opt = int(unused_argv[1])
    except Exception as e:
        print("USer option reading failed")
        print(e)

    if user_opt:
        print("User Option : ", user_opt)
        if user_opt == 1:
            print("###### Prediction Run ######")
    else:
        print("###### Training and Evaluation Run #######")

    # Load training and eval data
    # Create the Estimator
    furniture_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./tensorflow_models")

    if user_opt:
        predict_out = furniture_classifier.predict(input_fn=test_input_fn,
                                   yield_single_examples=True)
        pred_buffer = []
        for pred in predict_out:
            pred_buffer.append((int(pred["image_id"]), int(pred["classes"])))
        if len(pred_buffer) == 0:
            return
        SUBMISSION_FP.write("id,predicted\n")
        pred_buffer = sorted(pred_buffer, key=lambda x: x[0])
        for i in range(len(pred_buffer)):
            out_str = str(pred_buffer[i][0]) + "," + str(pred_buffer[i][1])
            SUBMISSION_FP.write(out_str)
            SUBMISSION_FP.write("\n")
    else:
        # Set up logging for predictions
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

        # Train the model
        furniture_classifier.train(
            input_fn=train_input_fn,
            steps=200,
            hooks=[logging_hook])

        # Evaluate the model and print results
        eval_results = furniture_classifier.evaluate(input_fn=val_input_fn)
        print(eval_results)


if __name__ == "__main__":
    TRAIN_LABEL_DF = pd.read_csv(TRAIN_LABEL_PATH)
    VAL_LABEL_DF = pd.read_csv(VAL_LABEL_PATH)
    TRAIN_SAMPLES = len(TRAIN_LABEL_DF)
    VAL_SAMPLES = len(VAL_LABEL_DF)
    with open(os.path.join(COMP_DATA_DIR, 'test.json')) as fp:
        test_json = json.loads(fp.read())
        TEST_SAMPLES = len(test_json['images'])
    print("# of training samples : " , TRAIN_SAMPLES)
    print("# of validation samples : ", VAL_SAMPLES)
    print("# of test samples : ", TEST_SAMPLES)
    SUBMISSION_FP, SUBMISSION_FPATH = getSubmissionFile()
    if SUBMISSION_FP is None:
        print("Not able to create submission file")
        sys.exit(0)

    # run tensorflow application
    tf.app.run()

    # cleanup
    try:
        if os.stat(SUBMISSION_FPATH).st_size == 0:
            try:
                print("removing submission file : because there was nothing in it")
                os.remove(SUBMISSION_FPATH)
            except:
                pass
    except FileNotFoundError as e:
        print("Submission file is not present")
    SUBMISSION_FP.close()
