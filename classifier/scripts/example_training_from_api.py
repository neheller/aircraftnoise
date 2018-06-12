
import json

from models.convnet import ConvNet
from servers.convnetserver import ConvNetServer
from adapters.macadapter import MACAdapter
from preprocessing.preprocessor import Preprocessor
from training.convnettrainer import ConvNetTrainer

import numpy as np
import tensorflow as tf

'''
CONFIGURATION
'''

# Example JSON file
EXAMPLE_FILE = '../raw_data/sample.json'

'''
Preprocessing
'''
DIMENSION = 37
# IDs of events in the first set that kill it
bad = []
'''
Adapter
'''
# Number of folds for k-fold cross-validation (decided during preprocessing)
FOLDS = None

'''
Server
'''
# These are intermediates created by preprocessing and used by network
# directory to get training, validation, and testing data from
INPUT_DIR = "training_intermediate"
# directory to write all log, predictions, and saved models to
# script will exit before training if this exists (to avoid overwriting)
OUTPUT_DIR = "training_out"

'''
Network
'''
# file location of weights to restore from (i.e. weights/model1.ckpt)
# I recommend you train from scratch - so set this to None
INITIAL_WEIGHTS = None
#INITIAL_WEIGHTS = 'checkpoints/cvd_model.ckpt'

'''
Trainer
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
# probability value to use for dropout
KEEP_PROB = 0.6
# training batch size
BATCH_SIZE = 2000
# step at which to log status at modulo 0
DISPLAY_STEP = 10
# The location in which to save the model
SAVE_NAME = "example_training.ckpt"



'''
SCRIPT
'''
# Only run if this is the main module to be run
if __name__ == '__main__':

    # JSON object returned from api_call
    # replace this with however you would like it to work in production
    json_data = json.load(open(EXAMPLE_FILE))
    # NOTE if events in json object have neither "aircraft" nor "community" fields
    # in they will be labeled as community for training - probably try to avoid this


    # build preprocessor
    ppr = Preprocessor()

    # Process raw data
    #X, Y, events_found = ppr.get_raw_data(DIMENSION, [RAW_FILE], bad)
    X, Y, events_found = ppr.get_from_json(DIMENSION, json_data)
    X, Y = ppr.remove_outliers(X, Y)
    X, Y = ppr.normalize(X, Y)
    # Shove all events into the "training" subdirectory
    trX, trY, teX, teY, vaX, vaY = ppr.partition_for_training(X, Y, 1.0, 0.0)
    # Store events in intermediate directory (will be deleted on subsequent trainings)
    ppr.store_training_partitions(trX, trY, teX, teY, vaX, vaY, INPUT_DIR)

    # build adapter
    adapter = MACAdapter(INPUT_DIR, DIMENSION, FOLDS)

    # build model
    convnet = ConvNet(DIMENSION)

    # build server
    server = ConvNetServer(adapter, OUTPUT_DIR,
                        batch_size = BATCH_SIZE,
                        verbose = True,
                        use=False)

    # build trainer
    trainer = ConvNetTrainer(convnet, server, EPOCHS, STEPS_PER_EPOCH,
                          optimizer = OPTIMIZER,
                          opt_kwargs = OPT_KWARGS,
                          keep_prob = KEEP_PROB,
                          batch_size = BATCH_SIZE,
                          display_step = DISPLAY_STEP)

    # initiate training
    trainer.train(
        restore_model = INITIAL_WEIGHTS,
        save_model = SAVE_NAME
    )
