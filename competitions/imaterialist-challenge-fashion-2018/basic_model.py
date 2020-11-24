from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import os
import pandas as pd
import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image
from random import randint
import numpy as np
import time
import sys

from settings import BASE_DIR, DATA_DIR, BASE_DIR_KAGGLE
from appsettings import COMPETITION_NAME,\
    ROOT_DATA_DIR, COMPETITION_DATA,\
    COMPETITION_FILE_LIST, TEST_DATA,\
    TRAIN_DATA, VAL_DATA, IMAGE_PATH,\
    LABEL_FILE

# globals
COMP_DIR=os.path.join(BASE_DIR_KAGGLE, "competitions", COMPETITION_NAME)
COMP_DATA_DIR=os.path.join(DATA_DIR, ROOT_DATA_DIR, COMPETITION_DATA)
TRAIN_IMAGE_PATH=os.path.join(DATA_DIR, ROOT_DATA_DIR, TRAIN_DATA, IMAGE_PATH)
TEST_IMAGE_PATH=os.path.join(DATA_DIR, ROOT_DATA_DIR, TEST_DATA, IMAGE_PATH)
VAL_IMAGE_PATH=os.path.join(DATA_DIR, ROOT_DATA_DIR, VAL_DATA, IMAGE_PATH)
TRAIN_LABEL_PATH=os.path.join(DATA_DIR, ROOT_DATA_DIR, TRAIN_DATA, LABEL_FILE)
TEST_LABEL_PATH=os.path.join(DATA_DIR, ROOT_DATA_DIR, TEST_DATA, LABEL_FILE)
VAL_LABEL_PATH=os.path.join(DATA_DIR, ROOT_DATA_DIR, VAL_DATA, LABEL_FILE)


# from scratchpad
IMAGE_X = 28
IMAGE_Y = 28
IMAGE_LAYER = 3
CLASS_MIN = 1
CLASS_MAX = 228
TRAIN_LABEL_DF = None
VAL_LABEL_DF = None
TRAIN_SAMPLES = 0
VAL_SAMPLES = 0
SUBMISSION_FP = None
SUBMISSION_FPATH = ""

MAX_SAMPLES=1000

#model globals
num_epochs = 1
batchSize = 10

TF_MODEL_DIR="./tensorflow_models"
TF_PRED_OUT_DIR="kaggle_submission"

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

def getMultiLabelBinarizer(lab_list, max_class=256):
    retval = [0]*max_class
    for n in lab_list:
        retval[n] = 1
    return retval



def getRandomSample(selected_sofar, panda_df, image_filepath):
    randsample = 1
    while True:
        randsample = randint(1, len(panda_df) - 1)
        if randsample in selected_sofar:
            continue
        try:
            image_location = os.path.join(image_filepath, str(randsample))
            img=mpimg.imread(image_location)
            break
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return randsample

def getRandomSampleTest(selected_sofar, maxdata, image_filepath):
    randsample = 1
    while True:
        randsample = randint(1, maxdata - 1)
        if randsample in selected_sofar:
            continue
        try:
            image_location = os.path.join(image_filepath, str(randsample))
            #img=mpimg.imread(image_location)
            img=Image.open(image_location)
            img=img.resize((IMAGE_X, IMAGE_Y))
            break
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return randsample

def getFilenamesLabels(panda_df, image_filepath, maxdata=1000):
    fnames = []
    flabels = []
    selected_so_far = []
    for _ in range(maxdata):
        randsample = getRandomSample(selected_so_far, panda_df, image_filepath)
        selected_so_far.append(randsample)
        fnames.append(os.path.join(image_filepath, str(randsample)))
        flabels.append(
            getMultiLabelBinarizer(
                [int(labnum) for labnum in panda_df['predicted'][randsample - 1].split()]
            ))
    return fnames, flabels

def verifyTestSample(image_id, image_filepath):
    try:
        image_location = os.path.join(image_filepath, str(image_id))
        # img=mpimg.imread(image_location)
        img = Image.open(image_location)
        img = img.resize((IMAGE_X, IMAGE_Y))
        return True
    except FileNotFoundError as e:
        print(e)
        return False
    except OSError as e:
        print(e)
        return False

def getFilenames(image_filepath, maxdata=1000):
    fnames = []
    selected_so_far = []
    image_ids = []
    for randsample in range(1, maxdata):
        if not verifyTestSample(randsample, image_filepath):
            raise ValueError
        selected_so_far.append(randsample)
        fnames.append(os.path.join(image_filepath, str(randsample)))
        image_ids.append(randsample)
    return fnames, image_ids

# Parse funciton for dataset input functio
def _read_py_function(filename, label):
    image_decoded = Image.open(filename.decode())
    image_resized = image_decoded.resize((IMAGE_X, IMAGE_Y))
    return image_resized, label

# Use standard TensorFlow operations to resize the image to a fixed shape.
def _resize_function(image_decoded, label):
    image_decoded.set_shape([IMAGE_X, IMAGE_Y, IMAGE_LAYER])
    #image_resized = tf.image.resize_images(image_decoded, [IMAGE_X, IMAGE_Y]) # most likely redundant
    return tf.cast(tf.convert_to_tensor(image_decoded), dtype=tf.float32), label


def dataset_input_fn(filenames, flabels):
    # A vector of filenames.
    filenames = tf.constant(filenames)
    # `labels[i]` is the label for the image in `filenames[i].
    class_labels = tf.constant(flabels)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, class_labels))
    # dataset = dataset.map(_parse_function)
    dataset = dataset.map(
        lambda filename, label: tuple(tf.py_func(
            _read_py_function, [filename, label], [tf.uint8, label.dtype])))
    dataset = dataset.map(_resize_function)
    dataset = dataset.shuffle(buffer_size=1000)
    batched_dataset = dataset.batch(batchSize)
    batched_dataset = batched_dataset.repeat(num_epochs)
    iterator = batched_dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return {"image_data" : features }, labels

def train_input_fn():
    filenames, flabels = getFilenamesLabels(TRAIN_LABEL_DF, TRAIN_IMAGE_PATH, maxdata=MAX_SAMPLES)
    return dataset_input_fn(filenames, flabels)

def val_input_fn():
    filenames, flabels = getFilenamesLabels(VAL_LABEL_DF, VAL_IMAGE_PATH, maxdata=MAX_SAMPLES)
    return dataset_input_fn(filenames, flabels)


def _read_py_test_function(filename, img_id):
    image_decoded = Image.open(filename.decode())
    image_resized = image_decoded.resize((IMAGE_X, IMAGE_Y))
    return image_resized, img_id

def _resize_test_function(image_decoded, img_id):
    image_decoded.set_shape([IMAGE_X, IMAGE_Y, IMAGE_LAYER])
    return tf.cast(tf.convert_to_tensor(image_decoded), dtype=tf.float32), img_id

def test_input_fn():
    filenames, image_ids = getFilenames(TEST_IMAGE_PATH, maxdata=10)
    # A vector of filenames.
    filenames = tf.constant(filenames)
    image_ids = tf.constant(image_ids)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, image_ids))
    dataset = dataset.map(
        lambda filename, imid: tuple(tf.py_func(
            _read_py_test_function, [filename, imid], [tf.uint8, tf.int32])))
    dataset = dataset.map(_resize_test_function)
    dataset = dataset.shuffle(buffer_size=1000)
    batched_dataset = dataset.batch(batchSize)
    batched_dataset = batched_dataset.repeat(num_epochs)
    iterator = batched_dataset.make_one_shot_iterator()
    filearray, fileid = iterator.get_next()
    return {
        "image_data" : filearray,
        "image_id" : fileid
    }

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["image_data"], [-1, IMAGE_X, IMAGE_Y, IMAGE_LAYER])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    #pool2_flat = tf.reshape(pool2, [-1, 128 * 128 * 64]) [512, 512]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=256)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "image_id" : features["image_id"],
            "classes" : tf.sigmoid(logits, name="sigmoid_tensor")
        }
    else:
        predictions = tf.sigmoid(logits, name="sigmoid_tensor")

    tf.summary.merge_all()
    if mode == tf.estimator.ModeKeys.PREDICT:
        pred_hook = tf.train.SessionRunHook()
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          prediction_hooks=[pred_hook])

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sigmoid_cross_entropy(labels, predictions)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions)}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

"""
def printModelOutput(op, buffer_dict):
    #print(op["image_id"], ",", np.max(op["classes"]))
    out_str = ""
    out_str += str(op["image_id"])
    out_str += ","
    count = 1
    for c in op["classes"]:
        if c > 0.01:
            out_str += str(count)
            out_str += " "
        count = count + 1
    print(out_str)
    SUBMISSION_FP.write(out_str)
    SUBMISSION_FP.write("\n")
"""

def printModelOutput(op, buffer_dict):
    #print(op["image_id"], ",", np.max(op["classes"]))
    out_str = ""
    key_str = int(op["image_id"])

    count = 1
    for c in op["classes"]:
        # TODO : 0.5 is threshold make it configurable
        if c > 0.5:
            out_str += str(count)
            out_str += " "
        count = count + 1
    print(out_str)
    buffer_dict[key_str] = out_str

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
            print("Prediction Run")
    else:
        print("Training and Evaluation Run")

    # Load training and eval data
    # Create the Estimator
    fashion_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./tensorflow_models")

    if user_opt:
        predict_out = fashion_classifier.predict(input_fn=test_input_fn,
                                   yield_single_examples=True)
        SUBMISSION_FP.write("id,predicted\n")
        buffer_dict = {}
        for pred in predict_out:
            printModelOutput(pred, buffer_dict)
        # TODO : 40000 is test sample make it a variable
        for i in range(1, 40000):
            try:
                class_str = buffer_dict[i]
                out_str = str(i) + "," + class_str
                SUBMISSION_FP.write(out_str)
                SUBMISSION_FP.write("\n")
            except:
                break


    else:
        # Set up logging for predictions
        tensors_to_log = {"probabilities": "sigmoid_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

        # Train the model
        fashion_classifier.train(
            input_fn=train_input_fn,
            steps=200,
            hooks=[logging_hook])


        # Evaluate the model and print results
        eval_results = fashion_classifier.evaluate(input_fn=val_input_fn)
        print(eval_results)



if __name__ == "__main__":
    TRAIN_LABEL_DF = pd.read_csv(TRAIN_LABEL_PATH)
    VAL_LABEL_DF = pd.read_csv(VAL_LABEL_PATH)
    TRAIN_SAMPLES = len(TRAIN_LABEL_DF)
    VAL_SAMPLES = len(VAL_LABEL_DF)
    #print(TRAIN_SAMPLES, VAL_SAMPLES)
    #print(BASE_DIR)
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
                print("removing submission file : becuase there was nothing in it")
                os.remove(SUBMISSION_FPATH)
            except:
                pass
    except FileNotFoundError as e:
        print("Submission file is not present")
    SUBMISSION_FP.close()
