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

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar10
import re

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
#                            """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_dir', './Adam_finetune_bias_tuning_lr_0.0001_ti_150000_ellipse_weight_decay_0.015/cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './Adam_finetune_bias_tuning_lr_0.0001_ti_150000_ellipse_weight_decay_0.015/cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  with tf.Session(config=config) as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # conv1_quan = tf.constant(0.15)
    # conv2_quan = tf.constant(0.08)
    # local3_quan = tf.constant(0.04)
    # local4_quan = tf.constant(0.06)
    # softmax_linear_quan = tf.constant(0.29)
    '''
    conv1_quan = tf.constant(0.15)
    conv2_quan = tf.constant(0.07)
    local3_quan = tf.constant(0.03)
    local4_quan = tf.constant(0.05)
    softmax_linear_quan = tf.constant(0.29)
    '''
    conv1_quan2 = tf.constant(0.15)
    conv2_quan2 = tf.constant(0.07)
    local3_quan2 = tf.constant(0.03)
    local4_quan2 = tf.constant(0.05)
    softmax_linear_quan2 = tf.constant(0.29)
    
    
    conv1_quan = tf.constant(0.08)
    conv2_quan = tf.constant(0.02)
    local3_quan = tf.constant(0.02)
    local4_quan = tf.constant(0.025)
    softmax_linear_quan = tf.constant(0.15)
    # sess.run(tf.Print(conv1_quan, [conv1_quan], 'conv1_quan'))

    for var in tf.trainable_variables():
        weights_pattern_conv1 = "conv1/weights$"
        weights_pattern_conv2 = "conv2/weights$"
        weights_pattern_local3 = "local3/weights$"
        weights_pattern_local4 = "local4/weights$"
        weights_pattern_softmax_linear = "local4/softmax_linear/weights$"
        # #
        '''
        if re.compile(weights_pattern_conv1).match(var.op.name):
          conv1_ones_shape = tf.ones(shape=tf.shape(var))
          sess.run(tf.assign(var, tf.where(tf.less(var, -tf.divide(conv1_quan, 2.0)), -conv1_quan * conv1_ones_shape,
                  tf.where(tf.less(var, tf.divide(conv1_quan, 2.0)), 0. * conv1_ones_shape, conv1_quan * conv1_ones_shape))))
        elif re.compile(weights_pattern_conv2).match(var.op.name):
          conv2_ones_shape = tf.ones(shape=tf.shape(var))
          sess.run(tf.assign(var, tf.where(tf.less(var, -tf.divide(conv2_quan, 2.0)), -conv2_quan * conv2_ones_shape,
                  tf.where(tf.less(var, tf.divide(conv2_quan, 2.0)), 0. * conv2_ones_shape, conv2_quan * conv2_ones_shape))))
        elif re.compile(weights_pattern_local3).match(var.op.name):
          local3_ones_shape = tf.ones(shape=tf.shape(var))
          sess.run(tf.assign(var, tf.where(tf.less(var, -tf.divide(local3_quan, 2.0)), -local3_quan * local3_ones_shape,
                  tf.where(tf.less(var, tf.divide(local3_quan, 2.0)), 0. * local3_ones_shape, local3_quan * local3_ones_shape))))
        elif re.compile(weights_pattern_local4).match(var.op.name):
          local4_ones_shape = tf.ones(shape=tf.shape(var))
          sess.run(tf.assign(var, tf.where(tf.less(var, -tf.divide(local4_quan, 2.0)), -local4_quan * local4_ones_shape,
                  tf.where(tf.less(var, tf.divide(local4_quan, 2.0)), 0. * local4_ones_shape, local4_quan * local4_ones_shape))))
        elif re.compile(weights_pattern_softmax_linear).match(var.op.name):
          softmax_linear_ones_shape = tf.ones(shape=tf.shape(var))
          sess.run(tf.assign(var, tf.where(tf.less(var, -tf.divide(softmax_linear_quan, 2.0)), -softmax_linear_quan * softmax_linear_ones_shape,
                  tf.where(tf.less(var, tf.divide(softmax_linear_quan, 2.0)), 0. * softmax_linear_ones_shape, softmax_linear_quan * softmax_linear_ones_shape))))
        '''
        if re.compile(weights_pattern_conv1).match(var.op.name):
          conv1_ones_shape = tf.ones(shape=tf.shape(var))
          sess.run(tf.assign(var, tf.where(tf.less(var, -tf.add(0.5*conv1_quan, 0.5*conv1_quan2)), -conv1_quan2 * conv1_ones_shape,
                  tf.where(tf.less(var, -tf.divide(conv1_quan, 2.0)), -conv1_quan * conv1_ones_shape, tf.where(tf.less(var, tf.divide(conv1_quan, 2.0)), 0. * conv1_ones_shape,
                  tf.where(tf.less(var, tf.add(0.5*conv1_quan, 0.5*conv1_quan2)), conv1_quan * conv1_ones_shape, conv1_quan2 * conv1_ones_shape))))))
        elif re.compile(weights_pattern_conv2).match(var.op.name):
          conv2_ones_shape = tf.ones(shape=tf.shape(var))
          sess.run(tf.assign(var, tf.where(tf.less(var, -tf.add(0.5*conv2_quan, 0.5*conv2_quan2)), -conv2_quan2 * conv2_ones_shape,
                  tf.where(tf.less(var, -tf.divide(conv2_quan, 2.0)), -conv2_quan * conv2_ones_shape, tf.where(tf.less(var, tf.divide(conv2_quan, 2.0)), 0. * conv2_ones_shape,
                  tf.where(tf.less(var, tf.add(0.5*conv2_quan, 0.5*conv2_quan2)), conv2_quan * conv2_ones_shape, conv2_quan2 * conv2_ones_shape))))))
        '''
        elif re.compile(weights_pattern_local3).match(var.op.name):
          local3_ones_shape = tf.ones(shape=tf.shape(var))
          sess.run(tf.assign(var, tf.where(tf.less(var, -tf.add(0.5*local3_quan, 0.5*local3_quan2)), -local3_quan2 * local3_ones_shape,
                  tf.where(tf.less(var, -tf.divide(local3_quan, 2.0)), -local3_quan * local3_ones_shape, tf.where(tf.less(var, tf.divide(local3_quan, 2.0)), 0. * local3_ones_shape,
                  tf.where(tf.less(var, tf.add(0.5*local3_quan, 0.5*local3_quan2)), local3_quan * local3_ones_shape, local3_quan2 * local3_ones_shape))))))
        elif re.compile(weights_pattern_local4).match(var.op.name):
          local4_ones_shape = tf.ones(shape=tf.shape(var))
          sess.run(tf.assign(var, tf.where(tf.less(var, -tf.add(0.5*local4_quan, 0.5*local4_quan2)), -local4_quan2 * local4_ones_shape,
                  tf.where(tf.less(var, -tf.divide(local4_quan, 2.0)), -local4_quan * local4_ones_shape, tf.where(tf.less(var, tf.divide(local4_quan, 2.0)), 0. * local4_ones_shape,
                  tf.where(tf.less(var, tf.add(0.5*local4_quan, 0.5*local4_quan2)), local4_quan * local4_ones_shape, local4_quan2 * local4_ones_shape))))))
                  '''
        elif re.compile(weights_pattern_softmax_linear).match(var.op.name):
          softmax_linear_ones_shape = tf.ones(shape=tf.shape(var))
          sess.run(tf.assign(var, tf.where(tf.less(var, -tf.add(0.5*softmax_linear_quan, 0.5*softmax_linear_quan2)), -softmax_linear_quan2 * softmax_linear_ones_shape,
                  tf.where(tf.less(var, -tf.divide(softmax_linear_quan, 2.0)), -softmax_linear_quan * softmax_linear_ones_shape, tf.where(tf.less(var, tf.divide(softmax_linear_quan, 2.0)), 0. * softmax_linear_ones_shape,
                  tf.where(tf.less(var, tf.add(0.5*softmax_linear_quan, 0.5*softmax_linear_quan2)), softmax_linear_quan * softmax_linear_ones_shape, softmax_linear_quan2 * softmax_linear_ones_shape))))))
        
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      # print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
      print('%s: precision before quantization @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    images, labels = cifar10.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)
    # saver.restore(sess, ckpt.model_checkpoint_path)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    #saver = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver(tf.trainable_variables())


    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
