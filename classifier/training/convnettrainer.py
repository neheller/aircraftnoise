import tensorflow as tf
import sys
import os
import numpy as np

'''
The object that encapsulates the training procedure and status
'''
class ConvNetTrainer(object):

    '''
    The object constructor
    '''
    # net:             the model object we are training
    # server:          the server object to use for all operating system
    #                  interactions
    # epochs:          number of epochs for training
    # steps_per_epoch: number of training steps per epoch of training
    # optimizer:       string name of the appropriate optimizer
    # opt_kwargs:      keyword arguments to accompany the optimizer
    # batch_size:    The size of each training batch to request
    # display_step:  Log a status every multiple of display step
    # keep_prob:     The dropout probability to use during training
    def __init__(self, net, server, epochs, steps_per_epoch,
                    optimizer = 'Adam',
                    opt_kwargs = {},
                    keep_prob = 1.0,
                    batch_size = 10,
                    display_step = 10):
        # keep pre-constructed objects
        self.net = net
        self.server = server

        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.display_step = display_step

        # parameters for training process
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        # variable that holds the global step of training
        self.global_step = tf.Variable(0)

        self.optimizer = self._get_optimizer(optimizer, opt_kwargs)


    '''
    Configure and return the appropriate optimizer for training
    '''
    # opt_type:   the string name of the optimizer to use (i.e. "Adam")
    # opt_kwargs: the keyword arguments to accompany the particular optimizer
    def _get_optimizer(self, opt_type, opt_kwargs):
        if opt_type == 'Adam':
            # get learning rate from kwargs
            lr = opt_kwargs.pop("learning_rate", 0.2)

            # Used to be exponential decay -- keeping the misnomer for now
            self.variable_learning_rate = tf.constant(lr, dtype=tf.float32)

            # Define optimizer objective
            optimizer = tf.train.AdamOptimizer(learning_rate=self.variable_learning_rate).minimize(self.net.cost)

            return optimizer

        else:
            print "Only Adam optimizer is currently supported - Exiting"
            sys.exit(0)

    '''
    The function that runs the training process for the network
    most execution time is spent here
    '''
    # restore_model: The path (relative) to a model checkpoint. If not None,
    #                the training starts with these weights
    def train(self, restore_model = None, save_model = None):
        # define the operation to initialize all variables
        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)

            # set model weights to saved values if called for
            if restore_model is not None:
                restore_abs = os.path.abspath(restore_model)
                self.server.log("restoring from: " + restore_abs, "info")
                self.net.restore(sess, restore_abs)

            # make prediction with initial weights
            val_x, val_dur, val_y = self.server.get_validation_batch(10000)
            val_preds, accuracy = sess.run((self.net.predictor, self.net.accuracy),
                                            feed_dict={
                                                self.net.x: val_x,
                                                self.net.durs: val_dur,
                                                self.net.y: val_y,
                                                self.net.keep_prob: 1.0
                                            })

            self.server.log("Weights initialized")
            self.server.log("Initial validation accuracy: " + str(accuracy), "info")

            # log the beginning of training
            self.server.log("Entering training loop")

            # Only save models better than 97% accuracy
            max_accuracy = 0.95
            # Iterate over all epochs
            for epoch in range(0, self.epochs):

                # Get batch for this epoch
                batch_x, batch_dur, batch_y = self.server.get_training_batch(self.batch_size)
                for step in range((epoch*self.steps_per_epoch),
                                  (epoch+1)*self.steps_per_epoch):


                    # Run optimization step
                    _, loss, lr, = sess.run((self.optimizer, self.net.w_ce,
                                           self.variable_learning_rate),
                                           feed_dict = {
                                                self.net.x: batch_x,
                                                self.net.durs: batch_dur,
                                                self.net.y: batch_y,
                                                self.net.keep_prob: self.keep_prob
                                           })

                    # Print step number if called for
                    if step % self.display_step == 0:
                        self.server.log("Step: " + str(step))

                        # Run prediction to get stats to display
                        val_x, val_dur, val_y = self.server.get_validation_batch(570)
                        val_preds, v_accuracy, v_f1 = sess.run((self.net.predictor, self.net.accuracy, self.net.f1),
                                                        feed_dict={
                                                            self.net.x: val_x,
                                                            self.net.durs: val_dur,
                                                            self.net.y: val_y,
                                                            self.net.keep_prob: 1.0
                                                        })

                        # Run prediction to get stats to display
                        t_accuracy, t_f1 = sess.run((self.net.accuracy, self.net.f1),
                                                        feed_dict={
                                                            self.net.x: batch_x,
                                                            self.net.durs: batch_dur,
                                                            self.net.y: batch_y,
                                                            self.net.keep_prob: 1.0
                                                        })

                        # log epoch
                        self.server.log("Step: " + str(step), "info")
                        self.server.log("Validation accuracy: " + str(v_accuracy), "info")
                        self.server.log("Validation f1: " + str(v_f1), "info")
                        self.server.log("Training accuracy:   " + str(t_accuracy), 'info')
                        self.server.log("Training f1: " + str(t_f1), "info")


                        if ((v_accuracy > max_accuracy) and (save_model is not None)):
                            max_accuracy = v_accuracy
                            self.server.save_weights(self.net, step, sess, custom_name=(str(v_accuracy) + save_model))
                # Run prediction after each training epoch
                val_preds, v_accuracy, v_f1 = sess.run((self.net.predictor, self.net.accuracy, self.net.f1),
                                                feed_dict={
                                                    self.net.x: val_x,
                                                    self.net.durs: val_dur,
                                                    self.net.y: val_y,
                                                    self.net.keep_prob: 1.0
                                                })

                # Run prediction after each training epoch
                t_accuracy, t_f1 = sess.run((self.net.accuracy, self.net.f1),
                                                feed_dict={
                                                    self.net.x: batch_x,
                                                    self.net.durs: batch_dur,
                                                    self.net.y: batch_y,
                                                    self.net.keep_prob: 1.0
                                                })

                # log epoch
                self.server.log("End of epoch " + str(epoch), "info")
                self.server.log("Validation accuracy: " + str(v_accuracy), "info")
                self.server.log("Validation f1: " + str(v_f1), "info")
                self.server.log("Training accuracy:   " + str(t_accuracy), 'info')
                self.server.log("Training f1: " + str(t_f1), "info")

            _, accuracy, precision, recall, f1 = sess.run((self.net.predictor, self.net.accuracy, self.net.precision, self.net.recall, self.net.f1),
                                            feed_dict={
                                                self.net.x: val_x,
                                                self.net.durs: val_dur,
                                                self.net.y: val_y,
                                                self.net.keep_prob: 1.0
                                            })
            if (save_model is not None):
                self.server.save_weights(self.net, step, sess, custom_name=save_model)


            return accuracy, precision, recall, f1
