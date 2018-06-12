import os
import sys
import numpy as np
import errno
import logging
import time

'''
The object that handles the bulk of the interactions with the operating system
This includes getting feed_dict data, storing predictions, and logging training
'''
class ConvNetServer(object):

    '''
    The server constructor
    '''
    # input_dir:        The directory to find all data for training/validation
    # output_dir:       The directory to store all data from training
    # batch_size:       The batch size to use for training (unless otherwise
    #                   specified at call-time)
    def __init__(self, adapter,
                    output_dir = "output",
                    batch_size = 1,
                    verbose = True,
                    use = False):

        # store adapter
        self.adapter = adapter


        self.use = use

        # make output path absolute
        self.output_dir = os.path.abspath(output_dir)
        self.predictions_dir = os.path.join(self.output_dir, "predictions")
        self.weights_dir = os.path.join(self.output_dir, "weights")

        # Check to make sure directory structure is valid
        self._check_io()

        # Set values for managing training
        self.batch_size = batch_size

        # configure the logging format
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")

        # create file handler
        if not self.use:
            log_filename = os.path.join(output_dir, "training" + str(time.time()) + ".log")
        else:
            log_filename = os.path.join(output_dir, "use" + str(time.time()) + ".log")
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

        # create console handler
        console_handler = logging.StreamHandler()
        if verbose:
            console_handler.setLevel(logging.INFO)
        else:
            console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(log_formatter)

        # add handlers to root logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # get adapter for image data
        self.adapter = adapter



    '''
    A function called by the constructor to make sure everything is in order
    for reading and writing to disk
    '''
    def _check_io(self):
        # Create structure for output. Error out if already exists to avoid
        # overwriting
        if not os.path.exists(self.output_dir):
            print "Output dir: %s does not exist - Creating it"%self.output_dir
            os.makedirs(self.output_dir)
            if not self.use:
                os.makedirs(self.predictions_dir)
                os.makedirs(self.weights_dir)
        else:
            print "Output dir: %s exists - ERROR"%self.output_dir
            sys.exit(0)

    '''
    A function to save the weights of the network to disk
    Calls corresponding net function with appropriate location
    '''
    # net:       The network object whose weights we're saving
    # iteration: The current iteration of training
    # session:   The current tensorflow session running
    def save_weights(self, net, iteration, session, custom_name = None):
        if (custom_name == None):
            save_path = os.path.join(self.weights_dir, 'step_' + str(iteration) + '.ckpt')
        else:
            save_path = os.path.join(self.weights_dir, custom_name)
        self.log("Saving model at " + save_path, "warning")
        net.save(session, save_path)

    '''
    A function to log information about the training process
    '''
    # message: the message to log
    def log(self, message, ltype="debug"):
        if (ltype=="debug"):
            self.logger.debug(message)
        elif (ltype=="warning"):
            self.logger.warning(message)
        else:
            self.logger.info(message)



    '''
    Set of functions which serve data to the training, validation, and testing
    procedure
    '''

    def get_training_batch(self, this_batch_size = None):
        if this_batch_size is None:
            this_batch_size = self.batch_size

        return self.adapter.get_batch(this_batch_size, "training")


    def get_validation_batch(self, this_batch_size = None):
        if this_batch_size is None:
            this_batch_size = self.batch_size

        return self.adapter.get_batch(this_batch_size, "validation")


    def get_testing_batch(self):
        return self.adapter.get_batch(None, "testing")
