# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from tensorflow.core.framework import summary_pb2
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 20
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.


FLAGS = None

def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32

def write_scalar_summary(summary_writer, tag, value, step):
  value = summary_pb2.Summary.Value(tag=tag, simple_value=float(value))
  summary = summary_pb2.Summary(value=[value])
  summary_writer.add_summary(summary, step)

def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath

def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialition.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=False,
                                                                seed=None,
                                                                dtype=tf.float32)
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable

def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels


def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])


def main(_):
  if FLAGS.self_test:
    print('Running self-test.')
    train_data, train_labels = fake_data(256)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
    num_epochs = 1
  else:
    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS
  train_size = train_labels.shape[0]

  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      data_type(),
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(
      data_type(),
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.global_variables_initializer().run()}
  # conv1_weights = tf.Variable(
  #     tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
  #                         stddev=0.1,
  #                         seed=SEED, dtype=data_type()), name='conv1_weights')
  #
  # conv2_weights = tf.Variable(tf.truncated_normal(
  #     [5, 5, 32, 64], stddev=0.1,
  #     seed=SEED, dtype=data_type()),name='conv2_weights')
  #
  # fc1_weights = tf.Variable(  # fully connected, depth 512.
  #     tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
  #                         stddev=0.1,
  #                         seed=SEED,
  #                         dtype=data_type()),name='fc1_weights')
  #
  # fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
  #                                               stddev=0.1,
  #                                               seed=SEED,
  #                                               dtype=data_type()), name='fc2_weights')

  conv1_weights = create_variable('conv1_weights', [5, 5, NUM_CHANNELS, 32])
  conv2_weights = create_variable('conv2_weights', [5, 5, 32, 64])
  fc1_weights = create_variable('fc1_weights', [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512])
  fc2_weights = create_variable('fc2_weights', [512, NUM_LABELS])

  #ini_conv1_weights = tf.Variable(create_variable('ini_conv1_weights', [5, 5, NUM_CHANNELS, 32]), trainable=False, name='ini_conv1_weights')
  #re_ini_conv1_weights = tf.reshape(ini_conv1_weights, [-1])
  #ini_conv1_mean, ini_conv1_variance = tf.nn.moments(re_ini_conv1_weights, [0])
  #ini_conv1_std = tf.sqrt(ini_conv1_variance)
  #print_ini_op0 = tf.Print(ini_conv1_std, [ini_conv1_std], 'ini_conv1_std')

  #ini_conv2_weights = tf.Variable(create_variable('ini_conv2_weights', [5, 5, 32, 64]), trainable=False, name='ini_conv2_weights')
  #re_ini_conv2_weights = tf.reshape(ini_conv2_weights, [-1])
  #ini_conv2_mean, ini_conv2_variance = tf.nn.moments(re_ini_conv2_weights, [0])
  #ini_conv2_std = tf.sqrt(ini_conv2_variance)
  #print_ini_op1 = tf.Print(ini_conv2_std, [ini_conv2_std], 'ini_conv2_std')


  #ini_fc1_weights = tf.Variable(create_variable('ini_fc1_weights', [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512]),trainable=False, name='ini_fc1_weights')
  #re_ini_fc1_weights = tf.reshape(ini_fc1_weights, [-1])
  #ini_fc1_mean, ini_fc1_variance = tf.nn.moments(re_ini_fc1_weights, [0])
  #ini_fc1_std = tf.sqrt(ini_fc1_variance)
  #print_ini_op2 = tf.Print(ini_fc1_std, [ini_fc1_std], 'ini_fc1_std')

  #ini_fc2_weights = tf.Variable(create_variable('ini_fc2_weights', [512, NUM_LABELS]),trainable=False, name='ini_fc2_weights')
  #re_ini_fc2_weights = tf.reshape(ini_fc2_weights, [-1])
  #ini_fc2_mean, ini_fc2_variance = tf.nn.moments(re_ini_fc2_weights, [0])
  #ini_fc2_std = tf.sqrt(ini_fc2_variance)
  #print_ini_op3 = tf.Print(ini_fc2_std, [ini_fc2_std], 'ini_fc2_std')



  conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()), name='conv1_biases')
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()), name='conv2_biases')
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()), name='fc1_biases')
  fc2_biases = tf.Variable(tf.constant(
      0.1, shape=[NUM_LABELS], dtype=data_type()), name='fc2_biases')

  #re_conv1_weights = tf.reshape(conv1_weights, [-1])
  #conv1_mean, conv1_variance = tf.nn.moments(re_conv1_weights, [0])
  #re_conv2_weights = tf.reshape(conv2_weights, [-1])
  #conv2_mean, conv2_variance = tf.nn.moments(re_conv2_weights, [0])
  #re_fc1_weights = tf.reshape(fc1_weights, [-1])
  #fc1_mean, fc1_variance = tf.nn.moments(re_fc1_weights, [0])
  #re_fc2_weights = tf.reshape(fc2_weights, [-1])
  #fc2_mean, fc2_variance = tf.nn.moments(re_fc2_weights, [0])

  #conv1_std = tf.sqrt(conv1_variance)
  #conv2_std = tf.sqrt(conv2_variance)
  #fc1_std = tf.sqrt(fc1_variance)
  #fc2_std = tf.sqrt(fc2_variance)

  #print_op0 = tf.Print(conv1_std, [conv1_std], 'conv1_std')
  #print_op1 = tf.Print(conv2_std, [conv2_std], 'conv2_std')
  #print_op2 = tf.Print(fc1_std, [fc1_std], 'fc1_std')
  #print_op3 = tf.Print(fc2_std, [fc2_std], 'fc2_std')
  #print_ini_op0 = tf.Print(ini_conv1_std, [ini_conv1_std], 'ini_conv1_std')
  #print_ini_op1 = tf.Print(ini_conv2_std, [ini_conv2_std], 'ini_conv2_std')
  #print_ini_op2 = tf.Print(ini_fc1_std, [ini_fc1_std], 'ini_fc1_std')
  #print_ini_op3 = tf.Print(ini_fc2_std, [ini_fc2_std], 'ini_fc2_std')


  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=train_labels_node, logits=logits))

  # L2 regularization for the fully connected parameters.
  # regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
  #                 tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))

  # add weights and biases of conv1 and conv2. added by Yandan:
  regularizers = (tf.nn.l2_loss(conv1_weights) + tf.nn.l2_loss(conv1_biases) +
                  tf.nn.l2_loss(conv2_weights) + tf.nn.l2_loss(conv2_biases) +
                  tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))

  # regularizers = (tf.nn.l1_loss(conv1_weights) + tf.nn.l1_loss(conv1_biases) +
  #                 tf.nn.l1_loss(conv2_weights) + tf.nn.l1_loss(conv2_biases) +
  #                 tf.nn.l1_loss(fc1_weights) + tf.nn.l1_loss(fc1_biases) +
  #                 tf.nn.l1_loss(fc2_weights) + tf.nn.l1_loss(fc2_biases))

  #quantification values: conv1: 0, 0.36, -0.36; conv2: 0, 0.07, -0.07; ip1: 0, 0.02, -0.02; ip2: 0, 0.18, -0.18
  # f1_conv1 = tf.sign(conv1_weights + 0.36)*(conv1_weights + 0.36)
  # f2_conv1 = tf.sign(conv1_weights ) * conv1_weights
  # f3_conv1 = tf.sign(conv1_weights - 0.36) * (conv1_weights - 0.36)
  #
  # f1_conv2 = tf.sign(conv2_weights + 0.07) * (conv2_weights + 0.07)
  # f2_conv2 = tf.sign(conv2_weights) * conv2_weights
  # f3_conv2 = tf.sign(conv2_weights - 0.07) * (conv2_weights - 0.07)
  #
  # f1_fc1 = tf.sign(fc1_weights + 0.02) * (fc1_weights + 0.02)
  # f2_fc1 = tf.sign(fc1_weights) * fc1_weights
  # f3_fc1 = tf.sign(fc1_weights - 0.02) * (fc1_weights - 0.02)
  #
  # f1_fc2 = tf.sign(fc2_weights + 0.18) * (fc2_weights + 0.18)
  # f2_fc2 = tf.sign(fc2_weights) * fc2_weights
  # f3_fc2 = tf.sign(fc2_weights - 0.18) * (fc2_weights - 0.18)

  n = tf.constant(2.5)
  conv1_std_co = tf.constant(0.14301154)
  conv2_std_co = tf.constant(0.028594673)
  fc1_std_co = tf.constant(0.016822236)
  fc2_std_co = tf.constant(0.076400243)
  
  #print_opn = tf.Print(n, [n], 'n')
  #print_op = tf.group(print_op0, print_op1, print_op2, print_op3, print_opn, print_ini_op0, print_ini_op1, print_ini_op2, print_ini_op3)

  conv1_quan = tf.multiply(n, conv1_std_co)
  conv2_quan = tf.multiply(n, conv2_std_co)
  fc1_quan = tf.multiply(n, fc1_std_co)
  fc2_quan = tf.multiply(n, fc2_std_co)

  f1_conv1 = tf.sign(conv1_weights + conv1_quan)*(conv1_weights + conv1_quan)
  f2_conv1 = tf.sign(conv1_weights ) * conv1_weights
  f3_conv1 = tf.sign(conv1_weights - conv1_quan) * (conv1_weights - conv1_quan)

  f1_conv2 = tf.sign(conv2_weights + conv2_quan) * (conv2_weights + conv2_quan)
  f2_conv2 = tf.sign(conv2_weights) * conv2_weights
  f3_conv2 = tf.sign(conv2_weights - conv2_quan) * (conv2_weights - conv2_quan)

  f1_fc1 = tf.sign(fc1_weights + fc1_quan) * (fc1_weights + fc1_quan)
  f2_fc1 = tf.sign(fc1_weights) * fc1_weights
  f3_fc1 = tf.sign(fc1_weights - fc1_quan) * (fc1_weights - fc1_quan)

  f1_fc2 = tf.sign(fc2_weights + fc2_quan) * (fc2_weights + fc2_quan)
  f2_fc2 = tf.sign(fc2_weights) * fc2_weights
  f3_fc2 = tf.sign(fc2_weights - fc2_quan) * (fc2_weights - fc2_quan)

  # conv1_regularizers = tf.where(tf.less(conv1_weights, -0.18), f1_conv1, tf.where(tf.less(conv1_weights, 0.18), f2_conv1, f3_conv1))
  # conv2_regularizers = tf.where(tf.less(conv2_weights, -0.035), f1_conv2, tf.where(tf.less(conv2_weights, 0.035), f2_conv2, f3_conv2))
  # fc1_regularizers = tf.where(tf.less(fc1_weights, -0.01), f1_fc1, tf.where(tf.less(fc1_weights, 0.01), f2_fc1, f3_fc1))
  # fc2_regularizers = tf.where(tf.less(fc2_weights, -0.09), f1_fc2, tf.where(tf.less(fc2_weights, 0.09), f2_fc2, f3_fc2))

  conv1_regularizers = tf.where(tf.less(conv1_weights, -tf.divide(conv1_quan, 2.0)), f1_conv1, tf.where(tf.less(conv1_weights, tf.divide(conv1_quan, 2.0)), f2_conv1, f3_conv1))
  conv2_regularizers = tf.where(tf.less(conv2_weights, -tf.divide(conv2_quan, 2.0)), f1_conv2, tf.where(tf.less(conv2_weights, tf.divide(conv2_quan, 2.0)), f2_conv2, f3_conv2))
  fc1_regularizers = tf.where(tf.less(fc1_weights, -tf.divide(fc1_quan, 2.0)), f1_fc1, tf.where(tf.less(fc1_weights, tf.divide(fc1_quan, 2.0)), f2_fc1, f3_fc1))
  fc2_regularizers = tf.where(tf.less(fc2_weights, -tf.divide(fc2_quan, 2.0)), f1_fc2, tf.where(tf.less(fc2_weights, tf.divide(fc2_quan, 2.0)), f2_fc2, f3_fc2))

  quantify_regularizers=(tf.reduce_sum(conv1_regularizers) +
                        tf.reduce_sum(conv2_regularizers) +
                        tf.reduce_sum(fc1_regularizers) +
                        tf.reduce_sum(fc2_regularizers))
  #quantify_regularizers=0
  # a = tf.constant(1)
  a = tf.Variable(1.,trainable=False, name='a')
  tf.summary.scalar(a.op.name, a)
  #a = tf.assign(a, tf.subtract(a, 0.00005))
  batch = tf.Variable(0, dtype=data_type())
  a = tf.assign(a, tf.add(tf.multiply(tf.divide(-1.0, (int(num_epochs * train_size) // BATCH_SIZE)),batch), 1))
  # a = tf.assign(a,tf.sqrt(1.0-tf.divide(tf.square(batch), tf.cast(tf.square(int(num_epochs * train_size) // BATCH_SIZE),tf.float32))))
  # a = tf.add(a, 0.0001)
  deformable_regularizers=a*regularizers+(1-a)*quantify_regularizers
  # deformable_regularizers = 0.5*regularizers + 0.5* quantify_regularizers

  # Add the regularization term to the loss.
  #loss += 5e-4 * regularizers
  # loss += 5e-4 * quantify_regularizers
  loss += 5e-4 * deformable_regularizers

  for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.

  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.975,                # Decay rate.
      staircase=True)
  tf.summary.scalar('learning_rate',learning_rate)
  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)

  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model(eval_data))

  summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
  # Build the summary operation from the last tower summaries.
  summary_op = tf.summary.merge(summaries)

  # Small utility function to evaluate a dataset by fe     eding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  summary_writer = tf.summary.FileWriter(
      './tb',
      graph=tf.get_default_graph())

  # Create a local session to run the training.
  start_time = time.time()
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.global_variables_initializer().run()
    print('Initialized!')
    # Loop through training steps.
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph it should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Run the optimizer to update weights.
      sess.run(optimizer, feed_dict=feed_dict)
      # print some extra information once reach the evaluation frequency
      if step % EVAL_FREQUENCY == 0:
        # fetch some extra nodes' data
        l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                      feed_dict=feed_dict)
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
        print('Validation error: %.1f%%' % error_rate(
            eval_in_batches(validation_data, sess), validation_labels))
        Validation_error = error_rate(
            eval_in_batches(validation_data, sess), validation_labels)
        write_scalar_summary(summary_writer, 'error', Validation_error, step)
        sys.stdout.flush()
      if step % 100 == 0:
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, step)
    #sess.run(print_op)

    # Finally print the result!
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    # tf.summary.scalar('test_error', test_error)
    print('Test error: %.1f%%' % test_error)
    if FLAGS.self_test:
      print('test_error', test_error)
      assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
          test_error,)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_fp16',
      default=False,
      help='Use half floats instead of full floats if True.',
      action='store_true')
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
