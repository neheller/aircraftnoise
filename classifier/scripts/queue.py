
import json
import os
import psycopg2
import psycopg2.extras
import shutil
import boto3
import signal
import time
import datetime

from models.convnet import ConvNet
from servers.convnetserver import ConvNetServer
from adapters.macadapter import MACAdapter
from preprocessing.preprocessor import Preprocessor

import numpy as np
import tensorflow as tf

class GracefulKiller:
    # http://stackoverflow.com/questions/18499497/how-to-process-sigterm-signal-gracefully
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True

class MachineLearning():
    def __init__(self):
        '''
        CONFIGURATION
        '''
        # Example JSON file
        self.MODEL = 1
        '''
        Preprocessing
        '''
        self.DIMENSION = 37
        # IDs of events in the first set that kill it
        self.bad = []
        '''
        Adapter
        '''
        # Number of folds for k-fold cross-validation (decided during preprocessing)
        self.FOLDS = None
        '''
        Server
        '''
        # These are intermediates created by preprocessing and used by network
        # directory to get training, validation, and testing data from
        self.INPUT_DIR = "use_test"
        # directory to write all log, predictions, and saved models to
        self.OUTPUT_DIR = "use_out"
        # dummy to make the network happy
        self.BATCH_SIZE = None
        '''
        Network
        '''
        # file location of weights to restore from (i.e. weights/model1.ckpt)
        self.INITIAL_WEIGHTS = 'checkpoints/cvd_model/cvd_model.ckpt'

    def db_open(self):
        self.conn = psycopg2.connect(
            "application_name=machine_learning" +
            " host=" + os.environ['PGHOST'] +
            " dbname=" + os.environ['PGDATABASE'] +
            " user=" + os.environ['PGUSER'] +
            " password=" + os.environ['PGPASSWORD'])

    def sqs_connect_to_queue(self):
        try:
            self.sqs = boto3.resource('sqs',
                        aws_access_key_id=os.environ['aws_access_key_id'],
                        aws_secret_access_key=os.environ['aws_secret_access_key'],
                        region_name=os.environ['region'])
            self.queue = self.sqs.get_queue_by_name(QueueName=os.environ['queue'])
        except Exception as e:
            self._catch_error(sys._getframe().f_code.co_name, e)

    def db_close(self):
        self.conn.close()

    def by_infile(self, infile):
        try:
            shutil.rmtree(self.OUTPUT_DIR)
        except:
            pass
        self.db_open()
        json_data = self.get_events_from_infile(infile)
        # build preprocessor
        ppr = Preprocessor()
        # Process raw data
        #X, Y, events_found = ppr.get_raw_data(DIMENSION, [RAW_FILE], bad)
        X, Y, events_found = ppr.get_from_json(self.DIMENSION, json_data)
        X, Y = ppr.remove_outliers(X, Y)
        X, Y = ppr.normalize(X, Y)
        trX, trY, teX, teY, vaX, vaY = ppr.partition_for_training(X, Y, 0.0, 1.0)
        ppr.store_training_partitions(trX, trY, teX, teY, vaX, vaY, self.INPUT_DIR)
        # build adapter
        adapter = MACAdapter(self.INPUT_DIR, self.DIMENSION, self.FOLDS)
        # build model
        convnet = ConvNet(self.DIMENSION)
        # build server
        server = ConvNetServer(adapter, self.OUTPUT_DIR,
                            batch_size = self.BATCH_SIZE,
                            verbose = True,
                            use=True)
        x, durs, _ = server.get_testing_batch()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            convnet.restore(sess, self.INITIAL_WEIGHTS)
            predictions = sess.run((convnet.predictor), feed_dict={
                convnet.x: x,
                convnet.durs: durs
            })
        # Get event ids
        _, _, ids = adapter.get_ids()
        results = [{"eventID": int(ids[i]), "ml": {"aircraftProbability": round(np.around(predictions[i][0],decimals=4),4), "model": self.MODEL}} for i in range(0, len(ids))]
        for result in results:
            self.insert_result_for_event(result)
        self.db_close()

    def get_events_from_infile(self, infile):
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        sql = "SELECT eventid::int, meta->'ehistory' as ehistory FROM macnoms.events WHERE infile = %s AND stime > '2017-01-01'"
        data = [infile]
        cur.execute(sql, data)
        results = cur.fetchall()
        return results

    def insert_result_for_event(self, result):
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        sql = "UPDATE macnoms.events set meta = jsonb_set(meta, '{ml}', %s) WHERE eventid = %s"
        data = [json.dumps(result['ml']), result['eventID']]
        self.conn.commit()
        cur.execute(sql, data)
'''
SCRIPT
'''
# Only run if this is the main module to be run
if __name__ == "__main__":

    q = MachineLearning()
    q.sqs_connect_to_queue()

    killer = GracefulKiller()

    def readQueue():
        jobs = q.queue.receive_messages()
        if len(jobs) > 0:
            for message in jobs:
                try:
                    message_body = json.loads(message.body)
                except ValueError:
                    print('Invalid json')
                    message_body = {}
                    message.delete()

                if 'job' in message_body:
                    d = datetime.datetime.now()
                    print(str(datetime.datetime.now()) + ': ' + message.body)
                    try:
                        data = message_body['job']
                        this_category = data['category']
                        this_operation = data['operation']
                    except Exception as e:
                        message.delete()
                        q._catch_error('queue.py', e)
                        return True

                    if this_category == 'machineLearning':
                        if this_operation == 'byInfile':
                            q.by_infile(data['inFile'])
                        else:
                            q._catch_error('main.py', 'Unknown operation (' + str(this_operation) + ')')
                        message.delete()

                    else:
                        q._catch_error('main.py', 'Unknown category (' + str(this_category) + ')')
                        message.delete()

                    print('---Done---')
                    return True
                else:
                    q._catch_error('main.py', 'JSON data does not contain the a job value')
                    message.delete()
                    return True
        else:
            return False

    while True:
        if killer.kill_now:  # Check for sigkill
            exit()
        if readQueue():
            pass
        else:
            time.sleep(60)  # Avoid too many queue polls
