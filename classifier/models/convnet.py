from __future__ import division
from collections import OrderedDict
import numpy as np
from math import ceil, floor

from models.layers import *

'''
The function that defines the set of computations that takes the input x
to the set of logits predicted for each event
'''
def build_convnet(x, durs, csize=3, ksize=2, dim=10):
    x_shape = tf.shape(x)
    batch_size = x_shape[0]

    height = 37
    width = dim

    # Variables for first convolution
    w1 = weight_variable([csize, csize, 1, 4], stddev=np.sqrt(2 / (csize**2 * 4)))
    b1 = bias_variable([4])

    # Variables for second convolution
    w2 = weight_variable([csize, csize, 4, 8], stddev=np.sqrt(2 / (csize**2 * 8)))
    b2 = bias_variable([8])


    # First convolution and pooling
    conv1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID')
    h_conv1 = tf.nn.relu(conv1 + b1)
    height = height - csize + 1
    width = width - csize + 1
    pool1 = tf.nn.max_pool(h_conv1, ksize=[1, ksize, ksize, 1], strides=[1, ksize, ksize, 1], padding='VALID')
    height = ceil(float((height - (ksize - 1))) / float(ksize))
    width = ceil(float((width - (ksize - 1))) / float(ksize))

    # Second convolution and pooling
    conv2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding='VALID')
    h_conv2 = tf.nn.relu(conv2 + b2)
    height = height - csize + 1
    width = width - csize + 1
    pool2 = tf.nn.max_pool(h_conv2, ksize=[1, ksize, ksize, 1], strides=[1, ksize, ksize, 1], padding='VALID')
    height = int(ceil(float((height - (ksize - 1))) / float(ksize)))
    width = int(ceil(float((width - (ksize - 1))) / float(ksize)))

    # Flat classifier input with duration
    flattened_features = tf.concat((tf.reshape(pool2, [batch_size, height*width*8]), durs), axis=1)

    # First dense layer
    s1 = tf.layers.dense(flattened_features, 40, activation=tf.nn.relu)

    # Second dense layer
    s2 = tf.layers.dense(s1, 15, activation=tf.nn.relu)

    # Third dense layer
    s3 = tf.layers.dense(s2, 2, activation=tf.nn.relu)

    return s3




'''
The object encapsulating the operations of the network
'''
class ConvNet(object):

    '''
    The constructor for the Network
    '''
    def __init__(self, dim):
        tf.reset_default_graph()

        self.dim = dim

        # the placeholder for the input
        self.x = tf.placeholder(tf.float32, shape=[None, 37, self.dim, 1], name="input")
        # the placeholder for the durations
        self.durs = tf.placeholder(tf.float32, shape=[None, 1], name="duration")
        # the placeholder for the output
        self.y = tf.placeholder(tf.float32, shape=[None, 2], name="labels")

        # the placeholder for the dropout keep probability
        self.keep_prob = tf.placeholder(tf.float32)

        # build network, return outputs, variables, and offset
        self.out = build_convnet(self.x, self.durs, dim=self.dim)

        # define cost computation
        self.cost = self._get_cost(self.out)

        # define computations for showing accuracy of the network
        self.predictor = softmax(self.out)
        self.correct = tf.equal(tf.argmax(self.predictor, 1),
                                tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

        # Utitlity tensors for computing the f1 score
        sums = tf.argmax(self.predictor, 1) + tf.argmax(self.y, 1)
        difs = tf.argmax(self.predictor, 1) - tf.argmax(self.y, 1)

        # Compute f1 score
        self.true_pos = tf.reduce_sum(tf.cast(tf.equal(sums, 2), tf.int32))
        self.false_pos = tf.reduce_sum(tf.cast(tf.equal(difs, 1), tf.int32))
        self.false_neg = tf.reduce_sum(tf.cast(tf.equal(difs, -1), tf.int32))

        self.precision = self.true_pos/(self.true_pos + self.false_pos)
        self.recall = self.true_pos/(self.true_pos + self.false_neg)

        self.f1 = 2*self.precision*self.recall/(self.precision + self.recall)


    # A function to test the validity of the computation graph
    def test_shp(self):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            x_dummy = np.zeros([200,37,self.dim,1])
            dur_dummy = np.zeros([200,1])
            tmp = sess.run((self.out), feed_dict= {
                self.x: x_dummy,
                self.durs: dur_dummy,
                self.keep_prob: 1
            })
            print tmp.shape

    '''
    The function that defines the loss of the network
    called from the constructor
    '''
    def _get_cost(self, logits, weight=50000.0):
        # compute loss
        self.tlogits = logits
        self.tlabels = self.y
        each_ce = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(
            logits = logits,
            labels = self.y
        ), (-1,1))
        pos = tf.slice(self.y, [0,1], [tf.shape(self.y)[0], 1])
        # get proportion of events that are community for class weighting
        ratio = 1-tf.reduce_mean(pos)
        # perform class weighting
        self.pwts = tf.multiply(pos,ratio)
        self.nwts = tf.multiply(tf.subtract(1.0, pos), 1-ratio)
        self.wts = tf.add(self.pwts, self.nwts)
        self.each_ce = each_ce
        self.w_ce = tf.multiply(each_ce, self.wts)
        # loss is the weighted average of cross-entropies
        loss = tf.reduce_mean(self.w_ce)
        return loss

    '''
    The function to make a prediction for each class for each pixel
    given some batch of input images
    '''
    # model_path: the location of the checkpoint in which the trained
    #             model was saved
    # x:          the tensor storing the input data to be predicted
    def predict(self, x, model_path=None):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # initialize all variables
            sess.run(init)

            # set weights to saved values
            if (model_path != None):
                self.restore(sess, model_path)

            y_emp = np.empty((x.shape[0], x.shape[1], x.shape[2], self.classes))
            prediction = sess.run(self.predictor, feed_dict={
                self.x: x,
                self.y: y_emp,
                self.keep_prob: 1.0
            })
        return prediction

    '''
    The function to save the current weights to a file in order to restore from
    them later
    '''
    # sess:       the current session with the desired variable values
    # model_path: the location in which to store the weights
    # RETURNS:    the location where it was saved
    def save(self, sess, model_path):
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    '''
    The function to restore a previous session's weights to the current session
    '''
    # sess:       the current session with the weights to be replaced
    # model_path: the location of the weights to restore
    # RETURNS:    None
    def restore(self, sess, model_path):
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
