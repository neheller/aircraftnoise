from models.convnet import ConvNet
from servers.convnetserver import ConvNetServer
from training.convnettrainer import ConvNetTrainer
from adapters.macadapter import MACAdapter

import numpy as np

'''
CONFIGURATION
'''

'''
Server
'''
# directory to get training, validation, and testing data from
INPUT_DIR = "devin"
# directory to write all log, predictions, and saved models to
OUTPUT_DIR = "devout"

'''
Network
'''


'''
Training
'''
# number of epochs to train for
EPOCHS = 1000
# number of training steps in each epoch
STEPS_PER_EPOCH = 250
# string name of optimizer to use
OPTIMIZER = "Momentum"
# keyword arguments for optimizer definition
#           learning_rate, default = 0.2
#           decay_rate,    default = 0.95
#           momentum,      default = 0.2
OPT_KWARGS = dict([("learning_rate",0.006), ("momentum",0.0)])

# file location of weights to restore from (i.e. weights/model1.ckpt)
INITIAL_WEIGHTS = './poor.ckpt'
# probability value to use for dropout
KEEP_PROB = 1.0
# training batch size
BATCH_SIZE = 400
# step at which to log status at modulo 0
DISPLAY_STEP = 5


'''
SCRIPT
'''
# Only run if this is the main module to be run
if __name__ == '__main__':

    # build adapter
    adapter = MACAdapter(INPUT_DIR)

    # build model
    convnet = ConvNet(10)

    # build server
    server = ConvNetServer(adapter, OUTPUT_DIR,
                        batch_size = BATCH_SIZE)

    # build trainer
    trainer = ConvNetTrainer(convnet, server, EPOCHS, STEPS_PER_EPOCH,
                          optimizer = OPTIMIZER,
                          opt_kwargs = OPT_KWARGS)

    convnet.test_shp()
