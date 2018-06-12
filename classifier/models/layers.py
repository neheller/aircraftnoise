from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf

'''
Functions to initialize variables
'''

def weight_variable(shape, stddev):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))


def softmax(mp):
    exponentials = tf.exp(mp)
    sums = tf.reduce_sum(exponentials, 1, keep_dims=True)
    return tf.div(exponentials, sums)
