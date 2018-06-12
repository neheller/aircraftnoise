from models.convnet import ConvNet
from servers.convnetserver import ConvNetServer
from training.convnettrainer import ConvNetTrainer
from adapters.macadapter import MACAdapter
from testing.convnettester import ConvNetTester

import numpy as np

'''
CONFIGURATION
'''

'''
Adapter
'''
# Number of folds for k-fold cross-validation (decided during preprocessing)
FOLDS = 10

'''
Server
'''
# directory to get training, validation, and testing data from
#INPUT_DIR = "10foldcv"
INPUT_DIR = "10foldcv"
# directory to write all log, predictions, and saved models to
OUTPUT_DIR = "cvout"

'''
Network
'''


'''
Training
'''
# number of epochs to train for
EPOCHS = 60
# number of training steps in each epoch
STEPS_PER_EPOCH = 85
# string name of optimizer to use
OPTIMIZER = "Adam"
# keyword arguments for optimizer definition
#           learning_rate, default = 0.2
OPT_KWARGS = dict([("learning_rate", 0.0004)])

# file location of weights to restore from (i.e. weights/model1.ckpt)
INITIAL_WEIGHTS = None
# probability value to use for dropout
KEEP_PROB = 0.6
# training batch size
BATCH_SIZE = 2000
# step at which to log status at modulo 0
DISPLAY_STEP = 10
# The interpolated dimensionality of each octave
DIMENSION = 37


'''
Tester
'''
# Number of trials to do for each fold (stats will be averaged)
TRIALS_PER_FOLD = 5


'''
SCRIPT
'''
# Only run if this is the main module to be run
if __name__ == '__main__':

    # build adapter
    adapter = MACAdapter(INPUT_DIR, DIMENSION, FOLDS)

    # build model
    convnet = ConvNet(DIMENSION)

    # build server
    server = ConvNetServer(adapter, OUTPUT_DIR,
                        batch_size = BATCH_SIZE,
                        verbose = False)

    # build trainer
    trainer = ConvNetTrainer(convnet, server, EPOCHS, STEPS_PER_EPOCH,
                          optimizer = OPTIMIZER,
                          opt_kwargs = OPT_KWARGS,
                          keep_prob = KEEP_PROB,
                          batch_size = BATCH_SIZE,
                          display_step = DISPLAY_STEP)


    # build tester
    tester = ConvNetTester(convnet, server, trainer)

    # initiate cross-validation
    tester.run_cross_validation(
       folds = FOLDS,
       trials_per_fold = TRIALS_PER_FOLD
    )
