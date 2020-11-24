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


def getRandomSample(selected_sofar, maxdata, image_filepath):
    rejected_so_far = []
    while True:
        if len(selected_sofar) + len(rejected_so_far) == maxdata:
            break
        randsample = randint(1, maxdata - 1)
        if randsample in selected_sofar:
            continue
        image_location = os.path.join(image_filepath, str(randsample))
        processed_location = image_location + ".jpeg"
        if os.path.exists(processed_location):
            return randsample
        if verifyImage(image_location):
            if rgb_path2path(image_location, processed_location, overwrite=False) == 0:
                return randsample
        rejected_so_far.append(randsample)
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
            flabels.append(int(label_df['predicted'][randsample - 1]))
    return image_ids, fnames, flabels


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def create_module_graph(module_spec):
    """Creates a graph and loads Hub Module into it.

    Args:
    module_spec: the hub.ModuleSpec for the image module being used.

    Returns:
    graph: the tf.Graph that was created.
    bottleneck_tensor: the bottleneck values output by the module.
    resized_input_tensor: the input images, resized as expected by the module.
    """
    height, width = hub.get_expected_image_size(module_spec)
    with tf.Graph().as_default() as graph:
        resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
        m = hub.Module(module_spec)
        bottleneck_tensor = m(resized_input_tensor)
        wants_quantization = any(node.op in FAKE_QUANT_OPS for node in graph.as_graph_def().node)
    return graph, bottleneck_tensor, resized_input_tensor, wants_quantization


def add_final_retrain_ops(class_count, final_tensor_name,
                          bottleneck_tensor, quantize_layer, is_training):
    # TODO : understand get_shape().as_list()
    batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()

    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor,
            shape=[batch_size, bottleneck_tensor_size],
            name='BottleneckInputPlaceholder')
        ground_truth_input = tf.placeholder(
            tf.int64, [batch_size], name='GroundTruthInput')

    # Organizing the following ops so they are easier to see in TensorBoard.
    layer_name = 'final_retrain_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal([bottleneck_tensor_size, class_count], stddev=0.001)
            layer_weights = tf.Variable(initial_value, name='final_weights')
            variable_summaries(layer_weights)
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    # The tf.contrib.quantize functions rewrite the graph in place for
    # quantization. The imported model graph has already been rewritten, so upon
    # calling these rewrites, only the newly added final layer will be
    # transformed.
    if quantize_layer:
        if is_training:
            tf.contrib.quantize.create_training_graph()
        else:
            tf.contrib.quantize.create_eval_graph()

    tf.summary.histogram('activations', final_tensor)
    # If this is an eval graph, we don't need to add loss ops or an optimizer.
    if not is_training:
        return None, None, bottleneck_input, ground_truth_input, final_tensor

    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels=ground_truth_input, logits=logits)

    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(tf_learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)
    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
            final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.

    Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

    Returns:
    Tuple of (evaluation step, prediction).
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, ground_truth_tensor)
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', evaluation_step)
        return evaluation_step, prediction


def cache_bottlenecks(sess, filenames, bottleneck_dir, image_data_tensor,
                      decoded_image_tensor, resized_input_tensor, bottleneck_tensor):
    bottleneck_count = 0
    for imgfile in filenames:
        get_or_create_bottleneck(sess, imgfile, bottleneck_dir, image_data_tensor, decoded_image_tensor,
                                 resized_input_tensor, bottleneck_tensor)
        bottleneck_count += 1
        if bottleneck_count % 100 == 0:
            tf.logging.info(str(bottleneck_count) + ' bottleneck files created.')


def get_or_create_bottleneck(sess, imgfile, bottleneck_dir, image_data_tensor, decoded_image_tensor,
                             resized_input_tensor, bottleneck_tensor):
    bottleneck_values = None
    bottleneck_path = os.path.join(bottleneck_dir, imgfile.split(os.sep)[-1])
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(sess, imgfile, image_data_tensor, decoded_image_tensor,
                               resized_input_tensor, bottleneck_tensor, bottleneck_path)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        tf.logging.warning('Invalid float found, recreating bottleneck')
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(sess, imgfile, image_data_tensor, decoded_image_tensor,
                               resized_input_tensor, bottleneck_tensor, bottleneck_path)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Allow exceptions to propagate here, since they shouldn't happen after a
        # fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def create_bottleneck_file(sess, imgfile, image_data_tensor, decoded_image_tensor,
                           resized_input_tensor, bottleneck_tensor, bottleneck_path):
    tf.logging.info('Creating bottleneck at ' + bottleneck_path)
    image_path = imgfile
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    processed_image_path = image_path + ".jpeg"
    if not tf.gfile.Exists(processed_image_path):
        tf.logging.fatal('Processed File does not exist %s', processed_image_path)
    image_data = tf.gfile.FastGFile(processed_image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(sess, image_data, image_data_tensor, decoded_image_tensor,
                                                    resized_input_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                     str(e)))
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def run_bottleneck_on_image(sess, image_data, image_data_tensor, decoded_image_tensor,
                            resized_input_tensor, bottleneck_tensor):
    # First decode the JPEG image, resize it, and rescale the pixel values.
    resized_input_values = sess.run(decoded_image_tensor,
                                    {image_data_tensor: image_data})
    # Then run it through the recognition network.
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def get_random_cached_bottlenecks(sess, fnames, flabels, batch_size,
                                  bottleneck_dir, image_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor):
    # TODO : make it random, although not necessary
    bottlenecks = []
    ground_truths = []
    filenames = []
    if batch_size <= 0:
        tf.logging.fatal('Invalid batch size %d', batch_size)
    for i in range(batch_size):
        if flabels is not None:
            ground_truths.append(flabels[i])
        filenames.append(fnames[i])
        bottleneck = get_or_create_bottleneck(sess, fnames[i], bottleneck_dir,
                                              decoded_image_tensor, image_data_tensor,
                                              bottleneck_tensor, resized_input_tensor)
        bottlenecks.append(bottleneck)
    return bottlenecks, ground_truths, filenames


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


def build_eval_session(module_spec, class_count):
    """Builds an restored eval session without train operations for exporting.

    Args:
    module_spec: The hub.ModuleSpec for the image module being used.
    class_count: Number of classes

    Returns:
    Eval session containing the restored eval graph.
    The bottleneck input, ground truth, eval step, and prediction tensors.
    """
    # If quantized, we need to create the correct eval graph for exporting.
    eval_graph, bottleneck_tensor, resized_input_tensor, wants_quantization = create_module_graph(module_spec)

    eval_sess = tf.Session(graph=eval_graph)
    with eval_graph.as_default():
        # Add the new layer for exporting.
        (_, _, bottleneck_input, ground_truth_input, final_tensor) = add_final_retrain_ops(
            class_count, tf_final_tensor_name, bottleneck_tensor, wants_quantization, is_training=False)

        # Now we need to restore the values from the training graph to the eval
        # graph.
        tf.train.Saver().restore(eval_sess, TF_CHECKPOINT_NAME)

        evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)

    return eval_sess, resized_input_tensor, bottleneck_input, ground_truth_input, evaluation_step, prediction


def save_graph_to_file(graph, graph_file_name, module_spec, class_count):
    """Saves an graph to file, creating a valid quantized one if necessary."""
    sess, _, _, _, _, _ = build_eval_session(module_spec, class_count)
    graph = sess.graph

    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [tf_final_tensor_name])
    with tf.gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())


def run_final_eval(train_session, fnames, flabels,
                   jpeg_data_tensor, decoded_image_tensor,
                   resized_image_tensor, bottleneck_tensor):
    """Runs a final evaluation on an eval graph using the test data set.

    Args:
      train_session: Session for the train graph with the tensors below.
      module_spec: The hub.ModuleSpec for the image module being used.
      class_count: Number of classes
      image_lists: OrderedDict of training images for each label.
      jpeg_data_tensor: The layer to feed jpeg image data into.
      decoded_image_tensor: The output of decoding and resizing the image.
      resized_image_tensor: The input node of the recognition graph.
      bottleneck_tensor: The bottleneck output layer of the CNN graph.
    """
    test_bottlenecks, test_ground_truth, test_filenames = get_random_cached_bottlenecks(train_session,
                                                                                        fnames,
                                                                                        flabels,
                                                                                        MINI_BATCH_SIZE,
                                                                                        tf_bottleneck_dir_test,
                                                                                        jpeg_data_tensor,
                                                                                        decoded_image_tensor,
                                                                                        resized_image_tensor,
                                                                                        bottleneck_tensor)


def export_model(module_spec, class_count, saved_model_dir):
    """Exports model for serving.
        Args:
        module_spec: The hub.ModuleSpec for the image module being used.
        class_count: The number of classes.
        saved_model_dir: Directory in which to save exported model and variables.
    """
    # The SavedModel should hold the eval graph.
    sess, in_image, _, _, _, _ = build_eval_session(module_spec, class_count)
    graph = sess.graph
    with graph.as_default():
        inputs = {'image': tf.saved_model.utils.build_tensor_info(in_image)}

        out_classes = sess.graph.get_tensor_by_name(tf_final_tensor_name + ':0')
        outputs = {
            'prediction': tf.saved_model.utils.build_tensor_info(out_classes)
        }

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        # Save out the SavedModel.
        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.
                DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    signature
            },
            legacy_init_op=legacy_init_op)
        builder.save()


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    class_count = tf_class_count

    _, train_filenames, train_flabels = getDatasetArrays(TRAIN_LABEL_DF, TRAIN_IMAGE_PATH,
                                             TRAIN_SAMPLES, max_samples=MAX_SAMPLES_TRAIN)
    _, validation_filenames, validation_flabels = getDatasetArrays(VAL_LABEL_DF, VAL_IMAGE_PATH,
                                             VAL_SAMPLES, max_samples=MAX_SAMPLES_VAL)
    test_image_ids, test_filenames, _ = getDatasetArrays(None, TEST_IMAGE_PATH,
                                               TEST_SAMPLES, max_samples=MAX_SAMPLES_TEST)

    # Set up the pre-trained graph.
    module_spec = hub.load_module_spec(TF_HUB_MODULE_URL)
    graph, bottleneck_tensor, resized_image_tensor, wants_quantization = create_module_graph(module_spec)

    # Add the new layer that we'll be training.
    with graph.as_default():
        (train_step, cross_entropy, bottleneck_input,
         ground_truth_input, final_tensor) = add_final_retrain_ops(class_count,
                                                                   tf_final_tensor_name,
                                                                   bottleneck_tensor,
                                                                   wants_quantization,
                                                                   is_training=True)
    with tf.Session(graph=graph) as sess:
        # Initialize all weights: for the module to their pretrained values,
        # and for the newly added retraining layer to random initial values.
        init = tf.global_variables_initializer()
        sess.run(init)

        # Set up the image decoding sub-graph.
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(module_spec)

        # We'll make sure we've calculated the 'bottleneck' image summaries and
        # cached them on disk.
        cache_bottlenecks(sess, train_filenames, tf_bottleneck_dir_train,
                          jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                          bottleneck_tensor)
        cache_bottlenecks(sess, test_filenames, tf_bottleneck_dir_test,
                          jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                          bottleneck_tensor)
        cache_bottlenecks(sess, validation_filenames, tf_bottleneck_dir_validation,
                          jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                          bottleneck_tensor)

        # Create the operations we need to evaluate the accuracy of our new layer.
        evaluation_step, _ = add_evaluation_step(final_tensor, ground_truth_input)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(tf_summaries_dir + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(tf_summaries_dir + '/validation')

        # Create a train saver that is used to restore values into an eval graph
        # when exporting models.
        train_saver = tf.train.Saver()

        # Run the training for as many cycles as requested
        for i in range(tf_training_steps):
            # Get a batch of input bottleneck values, either calculated fresh every
            # time or from the cache stored on disk.
            (train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(sess,
                                                                                       train_filenames,
                                                                                       train_flabels,
                                                                                       MINI_BATCH_SIZE,
                                                                                       tf_bottleneck_dir_train,
                                                                                       jpeg_data_tensor,
                                                                                       decoded_image_tensor,
                                                                                       resized_image_tensor,
                                                                                       bottleneck_tensor)

            # Feed the bottlenecks and ground truth into the graph, and run a training
            # step. Capture training summaries for TensorBoard with the `merged` op.
            train_summary, _ = sess.run(
                [merged, train_step],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)

            # Every so often, print out how well the graph is training.
            is_last_step = (i + 1 == tf_training_steps)
            if (i % tf_eval_step_interval) == 0 or is_last_step:
                # TODO add evaluation step
                train_accuracy, cross_entropy_value = sess.run(
                    [evaluation_step, cross_entropy],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth})
                tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                                (datetime.now(), i, train_accuracy * 100))
                tf.logging.info('%s: Step %d: Cross entropy = %f' %
                                (datetime.now(), i, cross_entropy_value))
                # moving averages being updated by the validation set, though in
                # practice this makes a negligable difference.
                (validation_bottlenecks,
                 validation_ground_truth, _) = get_random_cached_bottlenecks(sess,
                                                                             validation_filenames,
                                                                             validation_flabels,
                                                                             MINI_BATCH_SIZE,
                                                                             tf_bottleneck_dir_validation,
                                                                             jpeg_data_tensor,
                                                                             decoded_image_tensor,
                                                                             resized_image_tensor,
                                                                             bottleneck_tensor)

                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                validation_summary, validation_accuracy = sess.run(
                    [merged, evaluation_step],
                    feed_dict={bottleneck_input: validation_bottlenecks,
                               ground_truth_input: validation_ground_truth})
                validation_writer.add_summary(validation_summary, i)
                tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                                (datetime.now(), i, validation_accuracy * 100,
                                len(validation_bottlenecks)))

            # Store intermediate results
            intermediate_frequency = tf_intermediate_store_frequency
            if intermediate_frequency > 0 and (i % intermediate_frequency == 0) and i > 0:
                # If we want to do an intermediate save, save a checkpoint of the train
                # graph, to restore into the eval graph.
                train_saver.save(sess, TF_CHECKPOINT_NAME)
                intermediate_file_name = os.path.join(tf_intermediate_output_graphs_dir,
                                                      'intermediate_' + str(i) + '.pb')
                tf.logging.info('Save intermediate result to : ' +
                                intermediate_file_name)
                save_graph_to_file(graph, intermediate_file_name, module_spec,
                                   class_count)

        # After training is complete, force one last save of the train checkpoint.
        train_saver.save(sess, TF_CHECKPOINT_NAME)
        # We've completed all our training, so run a final test evaluation on
        # some new images we haven't used before.
        run_final_eval(sess, test_filenames, None,
                       jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                       bottleneck_tensor)
        # Write out the trained graph and labels with the weights stored as
        # constants.
        tf.logging.info('Save final result to : ' + tf_output_graph)
        if wants_quantization:
            tf.logging.info('The model is instrumented for quantization with TF-Lite')
        save_graph_to_file(graph, os.path.join(tf_output_graph, "output_graph"), module_spec, class_count)
        export_model(module_spec, class_count, tf_saved_model_dir)


if __name__ == '__main__':

    # create all directories
    dir_to_check = [TF_MODEL_DIR, tf_bottleneck_dir, tf_summaries_dir,
                    tf_intermediate_output_graphs_dir, tf_bottleneck_dir_validation,
                    tf_bottleneck_dir_train, tf_bottleneck_dir_test, tf_output_graph]
    for dirr in dir_to_check:
        if not os.path.exists(dirr):
            os.makedirs(dirr)

    TRAIN_LABEL_DF = pd.read_csv(TRAIN_LABEL_PATH)
    VAL_LABEL_DF = pd.read_csv(VAL_LABEL_PATH)
    TRAIN_SAMPLES = len(TRAIN_LABEL_DF)
    VAL_SAMPLES = len(VAL_LABEL_DF)
    with open(os.path.join(COMP_DATA_DIR, 'test.json')) as fp:
        test_json = json.loads(fp.read())
        TEST_SAMPLES = len(test_json['images'])
    print("# of training samples : ", TRAIN_SAMPLES)
    print("# of validation samples : ", VAL_SAMPLES)
    print("# of test samples : ", TEST_SAMPLES)

    tf.app.run()
