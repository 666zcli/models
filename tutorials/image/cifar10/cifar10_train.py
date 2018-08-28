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

"""A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import re
import math

import tensorflow as tf

import cifar10
import os

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
#                            """Directory where to write event logs """
#                            """and checkpoint.""")
tf.app.flags.DEFINE_string('train_dir', './Adam_finetune_bias_tuning_lr_0.00005_ti_150000_ellipse/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 150000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_float('weight_decay', 0.002,
                            """Decay to learn quantized weights.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 1,
                            """How often to log results to the console.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    # loss = cifar10.loss(logits, labels)
    cross_entropy, l2_loss = cifar10.loss(logits, labels)

    # n = tf.constant(2.5)
    # n = tf.constant(1.0)
    # conv1_std_co = tf.constant(0.326984)
    # conv2_std_co = tf.constant(0.099911)
    # local3_std_co = tf.constant(0.010653)
    # local4_std_co = tf.constant(0.015261)
    # softmax_linear_std_co = tf.constant(0.222937)

    # conv1_quan = tf.multiply(n, conv1_std_co)
    # conv2_quan = tf.multiply(n, conv2_std_co)
    # local3_quan = tf.multiply(n, local3_std_co)
    # local4_quan = tf.multiply(n, local4_std_co)
    # softmax_linear_quan = tf.multiply(n, softmax_linear_std_co)

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
    
    conv1_quan = tf.constant(0.075)
    conv2_quan = tf.constant(0.035)
    local3_quan = tf.constant(0.015)
    local4_quan = tf.constant(0.025)
    softmax_linear_quan = tf.constant(0.15)

    #mytrainable_list = []
    for var in tf.trainable_variables():
    #    # mytrainable_list.append(var)
        weights_pattern_conv1 = ".*conv1/weights$"
        weights_pattern_conv2 = ".*conv2/weights$"
        weights_pattern_local3 = ".*local3/weights$"
        weights_pattern_local4 = ".*local4/weights$"
        weights_pattern_softmax_linear = ".*local4/softmax_linear/weights$"
    #    bias_pattern = re.compile("(.*conv1/biases$)|(.*conv2/biases$)|(.*local3/biases$)|(.*local4/biases$)|(.*local4/softmax_linear/biases$)")
        if re.compile(weights_pattern_conv1).match(var.op.name):
          conv1_weights = var
    #      # mytrainable_list.append(var)
        elif re.compile(weights_pattern_conv2).match(var.op.name):
          conv2_weights = var
    #      # mytrainable_list.append(var)
        elif re.compile(weights_pattern_local3).match(var.op.name):
          local3_weights = var
    #      # mytrainable_list.append(var)
        elif re.compile(weights_pattern_local4).match(var.op.name):
          local4_weights = var
    #      # mytrainable_list.append(var)
        elif re.compile(weights_pattern_softmax_linear).match(var.op.name):
          softmax_linear_weights = var
    #      # mytrainable_list.append(var)
    #    elif bias_pattern.match(var.op.name):
    #      mytrainable_list.append(var)
    #    else:
    #      raise RuntimeError('Some variables are not matched!!!')
    #tf.add_to_collection('mytrainable_list', mytrainable_list)

    f1_conv1 = tf.sign(conv1_weights + 2*conv1_quan) * (conv1_weights + 2*conv1_quan)
    f2_conv1 = tf.sign(conv1_weights + conv1_quan) * (conv1_weights + conv1_quan)
    f3_conv1 = tf.sign(conv1_weights) * conv1_weights
    f4_conv1 = tf.sign(conv1_weights - conv1_quan) * (conv1_weights - conv1_quan)
    f5_conv1 = tf.sign(conv1_weights - 2*conv1_quan) * (conv1_weights - 2*conv1_quan)
    
    '''
    f1_conv2 = tf.sign(conv2_weights + conv2_quan) * (conv2_weights + conv2_quan)
    f2_conv2 = tf.sign(conv2_weights) * conv2_weights
    f3_conv2 = tf.sign(conv2_weights - conv2_quan) * (conv2_weights - conv2_quan)
    f1_local3 = tf.sign(local3_weights + local3_quan) * (local3_weights + local3_quan)
    f2_local3 = tf.sign(local3_weights) * local3_weights
    f3_local3 = tf.sign(local3_weights - local3_quan) * (local3_weights - local3_quan)
    f1_local4 = tf.sign(local4_weights + local4_quan) * (local4_weights + local4_quan)
    f2_local4 = tf.sign(local4_weights) * local4_weights
    f3_local4 = tf.sign(local4_weights - local4_quan) * (local4_weights - local4_quan)
    f1_softmax_linear = tf.sign(softmax_linear_weights + softmax_linear_quan) * (softmax_linear_weights + softmax_linear_quan)
    f2_softmax_linear = tf.sign(softmax_linear_weights) * softmax_linear_weights
    f3_softmax_linear = tf.sign(softmax_linear_weights - softmax_linear_quan) * (softmax_linear_weights - softmax_linear_quan)
    '''
    
    f1_conv2 = tf.sign(conv2_weights + 2*conv2_quan) * (conv2_weights + 2*conv2_quan)
    f2_conv2 = tf.sign(conv2_weights + conv2_quan) * (conv2_weights + conv2_quan)
    f3_conv2 = tf.sign(conv2_weights) * conv2_weights
    f4_conv2 = tf.sign(conv2_weights - conv2_quan) * (conv2_weights - conv2_quan)
    f5_conv2 = tf.sign(conv2_weights - 2*conv2_quan) * (conv2_weights - 2*conv2_quan)
    
    f1_local3 = tf.sign(local3_weights + 2*local3_quan) * (local3_weights + 2*local3_quan)
    f2_local3 = tf.sign(local3_weights + local3_quan) * (local3_weights + local3_quan)
    f3_local3 = tf.sign(local3_weights) * local3_weights
    f4_local3 = tf.sign(local3_weights - local3_quan) * (local3_weights - local3_quan)
    f5_local3 = tf.sign(local3_weights - local3_quan) * (local3_weights - 2*local3_quan)
    
    f1_local4 = tf.sign(local4_weights + 2*local4_quan) * (local4_weights + 2*local4_quan)
    f2_local4 = tf.sign(local4_weights + local4_quan) * (local4_weights + local4_quan)
    f3_local4 = tf.sign(local4_weights) * local4_weights
    f4_local4 = tf.sign(local4_weights - local4_quan) * (local4_weights - local4_quan)
    f5_local4 = tf.sign(local4_weights - local4_quan) * (local4_weights - 2*local4_quan)
    
    f1_softmax_linear = tf.sign(softmax_linear_weights + 2*softmax_linear_quan) * (softmax_linear_weights + 2*softmax_linear_quan)
    f2_softmax_linear = tf.sign(softmax_linear_weights + softmax_linear_quan) * (softmax_linear_weights + softmax_linear_quan)
    f3_softmax_linear = tf.sign(softmax_linear_weights) * softmax_linear_weights
    f4_softmax_linear = tf.sign(softmax_linear_weights - softmax_linear_quan) * (softmax_linear_weights - softmax_linear_quan)
    f5_softmax_linear = tf.sign(softmax_linear_weights - 2*softmax_linear_quan) * (softmax_linear_weights - 2*softmax_linear_quan)
    
    
    
    
    conv1_regularizers = tf.where(tf.less(conv1_weights, -tf.multiply(conv1_quan, 1.5)), f1_conv1,
                                  tf.where(tf.less(conv1_weights, -tf.divide(conv1_quan, 2.0)), f2_conv1, tf.where(tf.less(conv1_weights, tf.divide(conv1_quan, 2.0)), f3_conv1,
                                  tf.where(tf.less(conv1_weights, tf.multiply(conv1_quan, 1.5)), f4_conv1, f5_conv1))))
    '''
    conv2_regularizers = tf.where(tf.less(conv2_weights, -tf.divide(conv2_quan, 2.0)), f1_conv2,
                                  tf.where(tf.less(conv2_weights, tf.divide(conv2_quan, 2.0)), f2_conv2, f3_conv2))
    local3_regularizers = tf.where(tf.less(local3_weights, -tf.divide(local3_quan, 2.0)), f1_local3,
                                tf.where(tf.less(local3_weights, tf.divide(local3_quan, 2.0)), f2_local3, f3_local3))
    local4_regularizers = tf.where(tf.less(local4_weights, -tf.divide(local4_quan, 2.0)), f1_local4,
                                tf.where(tf.less(local4_weights, tf.divide(local4_quan, 2.0)), f2_local4, f3_local4))
    softmax_linear_regularizers = tf.where(tf.less(softmax_linear_weights, -tf.divide(softmax_linear_quan, 2.0)), f1_softmax_linear,
                                   tf.where(tf.less(softmax_linear_weights, tf.divide(softmax_linear_quan, 2.0)), f2_softmax_linear, f3_softmax_linear))
    '''
    
    conv2_regularizers = tf.where(tf.less(conv2_weights, -tf.multiply(conv2_quan, 1.5)), f1_conv2,
                                  tf.where(tf.less(conv2_weights, -tf.divide(conv2_quan, 2.0)), f2_conv2, tf.where(tf.less(conv2_weights, tf.divide(conv2_quan, 2.0)), f3_conv2,
                                  tf.where(tf.less(conv2_weights, tf.multiply(conv2_quan, 1.5)), f4_conv2, f5_conv2))))
    local3_regularizers = tf.where(tf.less(local3_weights, -tf.multiply(local3_quan, 1.5)), f1_local3,
                                  tf.where(tf.less(local3_weights, -tf.divide(local3_quan, 2.0)), f2_local3, tf.where(tf.less(local3_weights, tf.divide(local3_quan, 2.0)), f3_local3,
                                  tf.where(tf.less(local3_weights, tf.multiply(local3_quan, 1.5)), f4_local3, f5_local3))))
    local4_regularizers = tf.where(tf.less(local4_weights, -tf.multiply(local4_quan, 1.5)), f1_local4,
                                  tf.where(tf.less(local4_weights, -tf.divide(local4_quan, 2.0)), f2_local4, tf.where(tf.less(local4_weights, tf.divide(local4_quan, 2.0)), f3_local4,
                                  tf.where(tf.less(local4_weights, tf.multiply(local4_quan, 1.5)), f4_local4, f5_local4))))
    softmax_linear_regularizers = tf.where(tf.less(softmax_linear_weights, -tf.multiply(softmax_linear_quan, 1.5)), f1_softmax_linear,
                                  tf.where(tf.less(softmax_linear_weights, -tf.divide(softmax_linear_quan, 2.0)), f2_softmax_linear, tf.where(tf.less(softmax_linear_weights, tf.divide(softmax_linear_quan, 2.0)), f3_softmax_linear,
                                  tf.where(tf.less(softmax_linear_weights, tf.multiply(softmax_linear_quan, 1.5)), f4_softmax_linear, f5_softmax_linear))))
    quantify_regularizers = (tf.reduce_sum(conv1_regularizers)+
                             tf.reduce_sum(conv2_regularizers)+
                             tf.reduce_sum(local3_regularizers)+
                             tf.reduce_sum(local4_regularizers)+
                             tf.reduce_sum(softmax_linear_regularizers)
                             )

    # # a changes with a square root of cosine function
    # a = tf.Variable(1., trainable=False, name='a')
    # tf.summary.scalar(a.op.name, a)
    # PI = tf.constant(math.pi)
    # a = tf.assign(a, tf.sqrt(0.5*(1.0+tf.cos(tf.divide(PI,FLAGS.max_steps)*tf.cast(global_step,tf.float32)))+1e-8))

    # a changes with a straight line
    # a = tf.assign(a, tf.add(tf.multiply(tf.divide(-1.0, (int(num_epochs * train_size) // BATCH_SIZE)),batch), 1))

    # a changes with a ellipse and sets to 0 at the final 5000 steps (N is the final steps to be set to 0)
    # N = tf.constant(5000)
    # a = tf.cond(tf.less(global_step, tf.cast(FLAGS.max_steps - N, tf.int64)), lambda:tf.assign(a,tf.cast(tf.sqrt(1.0-tf.divide(tf.cast(tf.square(global_step),tf.int32), tf.square(FLAGS.max_steps))), tf.float32)),lambda:tf.assign(a, 0.))

    # a changes with a cosine function
    a = tf.Variable(1., trainable=False, name='a')
    tf.summary.scalar(a.op.name, a)
    PI = tf.constant(math.pi)
    a = tf.assign(a, 0.5 * (1.0 + tf.cos(tf.divide(PI, FLAGS.max_steps) * tf.cast(global_step, tf.float32))) + 1e-8)

    # a changes with a cosine function sets to 0 at the final 5000 steps (N is the final steps to be set to 0)
    # a = tf.Variable(1., trainable=False, name='a')
    # tf.summary.scalar(a.op.name, a)
    # N = tf.constant(5000)
    # PI = tf.constant(math.pi)
    # a = tf.cond(tf.less(global_step, tf.cast(FLAGS.max_steps - N, tf.int64)), lambda:tf.assign(a, 0.5 * (1.0 + tf.cos(tf.divide(PI, FLAGS.max_steps) * tf.cast(global_step, tf.float32))) + 1e-8) ,
    #             lambda: tf.assign(a, 0.))

    # b = tf.Variable(0.5, trainable=False, name='b')
    # tf.summary.scalar(b.op.name, b)
    # b = tf.assign(b, tf.random_uniform([], 0., 1.))
    # deformable_regularizers = tf.where(tf.less(b, a), l2_loss, quantify_regularizers)

    # DECAY = tf.constant(0.012)

    deformable_regularizers = a * l2_loss + (1 - a) * quantify_regularizers
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    # train_op = cifar10.train(loss, global_step)
    # total_loss = cross_entropy + 0.001 * l2_loss
    total_loss = cross_entropy+FLAGS.weight_decay*deformable_regularizers
    # total_loss = cross_entropy + 0.001 * quantify_regularizers
    train_op = cifar10.train(total_loss, global_step)

    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('DECAY*deformable_regularizers', tf.multiply(FLAGS.weight_decay, deformable_regularizers))

    # for var in tf.trainable_variables():
    #   pattern = ".*weights.*"
    #   if re.compile(pattern).match(var.op.name):
    #     tf.summary.histogram(var.op.name, var)


    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(total_loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, total_loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    # for var in tf.trainable_variables():
    #     tf.summary.histogram(var.op.name, var)


    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    print_op = tf.no_op()
    # for var in tf.trainable_variables():
    #     weights_pattern = ".*weights.*"
    #     if re.compile(weights_pattern).match(var.op.name):
    #         ini_mean, ini_variance = tf.nn.moments(tf.reshape(var, [-1]), [0])
    #         ini_std = tf.sqrt(ini_variance)
    #         print_var = tf.Print(var.op.name, [var.op.name, ini_std], var.op.name)
    #         # print_ini_op0 = tf.Print(ini_std, [ini_std], 'ini_std')
    #         print_op = tf.group(print_op, print_var)
    # with tf.control_dependencies([print_op]):
    #     train_op = tf.group(train_op, tf.no_op())

    #
    # if global_step % 1 == 0:
    #   # summary_str = sess.run(summary_op)
    #   with tf.control_dependencies([summary_op]):
    #     train_op = tf.group(train_op, tf.no_op())
      # summary_writer.add_summary(summary_str, global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    summary_writer = tf.summary.FileWriter(
        FLAGS.train_dir,
        graph=tf.get_default_graph())

    with tf.Session(config=config) as sess:
        # saver = tf.train.import_meta_graph('./tb_no_quantization_baseline_300000/cifar10_train/model.ckpt-300000.meta')
        sess.run(tf.global_variables_initializer())
        var_dic = {}
        _vars = tf.global_variables()
        for _var in _vars:
            pattern = re.compile("(.*conv1/weights$)|(.*conv2/weights$)|(.*local3/weights$)|(.*local4/weights$)|(.*local4/softmax_linear/weights$)|(.*conv1/biases$)|(.*conv2/biases$)|(.*local3/biases$)|(.*local4/biases$)|(.*local4/softmax_linear/biases$)|(.*MovingAverage$)")
            if pattern.match(_var.op.name) :
                _var_name = _var.op.name
                var_dic[_var_name] = _var
        saver = tf.train.Saver(var_dic)

        # saver.restore(sess, "./pretrain_baseline_0.872_lr_0.0002_wd_0.001_ti_500000/cifar10_train/model.ckpt-500000")
        # saver.restore(sess, "./Adam_finetune_conv1_lr_0.00005_wd_0.01_ti_150000_aL1/cifar10_train/model.ckpt-150000")
        # saver.restore(sess, "./Adam_finetune_conv1_lr_0.00005_wd_0.01_ti_150000_ellipse/cifar10_train/model.ckpt-150000")
        # saver.restore(sess, "./Adam_finetune_conv1_lr_0.00005_wd_0.01_ti_150000_Bernoulli/cifar10_train/model.ckpt-150000")
        # saver.restore(sess,"./Adam_finetune_freeze_conv1_conv2_0.006_lr_0.0001_ti_121000_Bernoulli_v2/cifar10_train/model.ckpt-121000")
        # saver.restore(sess,"./Adam_finetune_freeze_conv1_conv2_0.006_lr_0.0001_ti_150000_ellipse/cifar10_train/model.ckpt-150000")
        # saver.restore(sess,"./Adam_finetune_freeze_conv1_conv2_0.006_lr_0.0001_ti_121000_Bernoulli_v3/cifar10_train/model.ckpt-121000")
        # saver.restore(sess,"./Adam_finetune_freeze_conv12_local3_0.003_lr_0.0001_ti_121000_Bernoulli/cifar10_train/model.ckpt-121000")
        # saver.restore(sess,"./Adam_finetune_freeze_conv12_local3_0.002_lr_0.0001_ti_150000_ellipse/cifar10_train/model.ckpt-150000")
        # saver.restore(sess,"./Adam_finetune_freeze_conv12_local3_0.002_lr_0.0001_ti_121000_Bernoulli_v3/cifar10_train/model.ckpt-121000")
        # saver.restore(sess,"./Adam_finetune_freeze_conv12local3_local4_0.01_lr_0.00005_ti_121000_Bernoulli/cifar10_train/model.ckpt-121000")
        # saver.restore(sess,"./Adam_finetune_freeze_conv12local3_local4_0.004_lr_0.0001_ti_121000_Bernoulli_v3/cifar10_train/model.ckpt-121000")
        # saver.restore(sess,"./Adam_finetune_freeze_conv12local3_local4_0.004_lr_0.0001_ti_150000_ellipse/cifar10_train/model.ckpt-150000")
        # saver.restore(sess,"./Adam_finetune_freeze_conv12local3_local4_0.008_lr_0.00005_ti_121000_Bernoulli_v3/cifar10_train/model.ckpt-121000")
        # saver.restore(sess,"./Adam_finetune_freeze_conv12local34_softmax_0.002_lr_0.00005_ti_121000_Bernoulli_v3/cifar10_train/model.ckpt-121000")
        # saver.restore(sess,"./Adam_finetune_freeze_conv12local34_softmax_0.002_lr_0.00005_ti_150000_ellipse/cifar10_train/model.ckpt-150000")

        # saver.restore(sess, "./Adam_finetune_freeze_conv1_conv2_0.005_lr_0.0001_ti_121000_Bernoulli/cifar10_train/model.ckpt-121000")

        # Start the queue runners.
        coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        tf.train.start_queue_runners(sess=sess, coord=coord)
        saver = tf.train.Saver()
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')

        # coord.request_stop()
        # coord.join(threads)

        for step in range(FLAGS.max_steps+1):
        # for step in range(1):
            if step % 1000 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            _, mloss = sess.run([train_op, total_loss])
            print('step {}: total loss {}'.format(step, mloss))
            if step%1000==0:
                saver.save(sess, checkpoint_path, global_step=step)


    # with tf.train.MonitoredTrainingSession(
    #     save_summaries_steps=10,
    #     checkpoint_dir=FLAGS.train_dir,
    #     hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
    #            tf.train.NanTensorHook(total_loss),
    #            _LoggerHook()],
    #     config=config) as mon_sess:
    #   while not mon_sess.should_stop():
    #     mon_sess.run(train_op)

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()



if __name__ == '__main__':
  tf.app.run()
