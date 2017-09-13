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

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  middle_size1=200
  middle_size2=100
  middle_size3=10

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.truncated_normal([784, middle_size1],stddev=0.1))
  b = tf.Variable(tf.truncated_normal([middle_size1],stddev=0.1))
  y = tf.nn.relu(tf.matmul(x, W) + b)

  W2 = tf.Variable(tf.truncated_normal([middle_size1, middle_size2],stddev=0.1))
  b2 = tf.Variable(tf.truncated_normal([middle_size2],stddev=0.1))
  y2 = tf.nn.relu(tf.matmul(y, W2) + b2)

  W3 = tf.Variable(tf.truncated_normal([middle_size2, middle_size3],stddev=0.1))
  b3 = tf.Variable(tf.truncated_normal([middle_size3],stddev=0.1))
  y3 = tf.nn.relu(tf.matmul(y2, W3) + b3)

  # W4 = tf.Variable(tf.truncated_normal([middle_size3, 10],stddev=0.1))
  # b4 = tf.Variable(tf.truncated_normal([10],stddev=0.1))
  # y4 = tf.nn.relu(tf.matmul(y3, W4) + b4)

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])
  output=y3

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
  #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Test trained model
  correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Train
  for step in range(25000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if step%100==0:print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
