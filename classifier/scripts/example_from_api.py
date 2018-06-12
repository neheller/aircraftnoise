
import json

from models.convnet import ConvNet
from servers.convnetserver import ConvNetServer
from adapters.macadapter import MACAdapter
from preprocessing.preprocessor import Preprocessor

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
INPUT_DIR = "use_test"
# directory to write all log, predictions, and saved models to
OUTPUT_DIR = "use_out"
# dummy to make the network happy
BATCH_SIZE = None

'''
Network
'''
# file location of weights to restore from (i.e. weights/model1.ckpt)
INITIAL_WEIGHTS = 'checkpoints/cvd_model/cvd_model.ckpt'

'''
SCRIPT
'''
# Only run if this is the main module to be run
if __name__ == '__main__':

    # JSON object returned from api_call
    # replace this with however you would like it to work in production
    json_data = json.load(open(EXAMPLE_FILE))

    # build preprocessor
    ppr = Preprocessor()

    # Process raw data
    #X, Y, events_found = ppr.get_raw_data(DIMENSION, [RAW_FILE], bad)
    X, Y, events_found = ppr.get_from_json(DIMENSION, json_data)
    X, Y = ppr.remove_outliers(X, Y)
    X, Y = ppr.normalize(X, Y)
    trX, trY, teX, teY, vaX, vaY = ppr.partition_for_training(X, Y, 0.0, 1.0)
    ppr.store_training_partitions(trX, trY, teX, teY, vaX, vaY, INPUT_DIR)

    # build adapter
    adapter = MACAdapter(INPUT_DIR, DIMENSION, FOLDS)

    # build model
    convnet = ConvNet(DIMENSION)

    # build server
    server = ConvNetServer(adapter, OUTPUT_DIR,
                        batch_size = BATCH_SIZE,
                        verbose = True,
                        use=True)

    x, durs, _ = server.get_testing_batch()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        convnet.restore(sess, INITIAL_WEIGHTS)

        predictions = sess.run((convnet.predictor), feed_dict={
            convnet.x: x,
            convnet.durs: durs
        })

    # Get event ids
    _, _, ids = adapter.get_ids()

    ret = [{"eventID": ids[i], "aircraftProbability": predictions[i][0]} for i in range(0, len(ids))]

    # Encode and send the labels back here
    print ret

    # # Display aircraft probability for each ID
    # for i in range(0, len(ids)):
    #     server.log(("Event %d: aircraft probability %.3f"%(
    #             ids[i], predictions[i][0])), "info")
