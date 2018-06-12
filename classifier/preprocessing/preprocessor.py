import numpy as np
import tensorflow as tf
import csv
import json
import math
import sys
import random
import os

from event2d import Event2D

class Preprocessor:

    # Contructor
    # does nothing atm
    def __init__(self):
        "nothing to be done"

    # Utility function of get_raw_data
    # adds event to the full numpy array
    def _encodenpy(self, e, x, y):
        return np.concatenate((x, e.to_array()), axis=0),  np.concatenate((y, np.array([[e.id, e.label]])), axis=0)

    # Utility function of get_raw_data
    # converts each line in input file into an event object
    # then encodes the event as an array with above function
    def _parse(self, rf, dim, bad):
        X = np.zeros((0,dim*37+1))
        Y = np.zeros((0,2))
        has_colnames = csv.Sniffer().has_header(rf.read(1024))
        rf.seek(0)
        reader = csv.reader(rf)
        if has_colnames:
            next(reader)
        num_rows = 0
        for row in reader:
            num_rows = num_rows + 1
            if int(row[0]) not in bad:
                event = Event2D(row, dim)
                if event.flag != 1:
                    X, Y = self._encodenpy(event, X, Y)

        return X, Y, num_rows


    # called by main
    # Stores raw data as an array with specified dimensionality*37 and durations
    # concatenated
    # returns tuple of input data and output data
    def get_from_json(self, dim, input_data):
        # Instantiate empty arrays for data
        X = np.zeros((0,dim*37+1))
        Y = np.zeros((0,2))

        events_found = len(input_data)

        # No need to parse since this is done
        for dat in input_data:
            event = Event2D(dat, dim, src='api')
            X, Y = self._encodenpy(event, X, Y)

        return X, Y, events_found


    # called by main
    # Stores raw data as an array with specified dimensionality*37 and durations
    # concatenated
    # returns tuple of input data and output data
    def get_raw_data(self, dim, input_files, bad):
        # Instantiate empty arrays for data
        X = np.zeros((0,dim*37+1))
        Y = np.zeros((0,2))

        events_found = 0

        for fil in input_files:
            print "Reading from " + fil
            rf = open(fil, 'rb')
            tmpx, tmpy, n_rows = self._parse(rf, dim, bad)
            events_found = events_found + n_rows
            if (tmpx is not None):
                X = np.concatenate((X, tmpx), axis=0)
                Y = np.concatenate((Y, tmpy), axis=0)

        return X, Y, events_found


    # Called by main
    # removes invalid and outlying events from dataset
    def remove_outliers(self, X, Y):
        Xshp1 = X.shape[1]
        Xret = np.zeros((0,Xshp1), dtype=np.float32)
        Yret = np.zeros((0,2), dtype=np.float32)
        i = 0
        for lin in X:
            if not ((np.isnan(lin).any()) or (np.max(lin) > 1e+4) or (np.min(lin) < -1e+4)):
                Xret = np.concatenate((Xret, np.reshape(X[i,:], (1,Xshp1))), axis=0)
                Yret = np.concatenate((Yret, np.reshape(Y[i,:], (1,2))), axis=0)
            i = i + 1
        return Xret, Yret


    # Called by main
    # normalizes the data to have mean zero
    def normalize(self, X, Y):
        mean_duration = np.mean(X[:,-1])
        mean_intensity = np.mean(X[:,0:-1])

        print "Mean duration before normalization: ", mean_duration
        print "Mean intensity before normalization:", mean_intensity
        print

        X[:,-1] = X[:,-1] - mean_duration
        X[:,0:-1] = X[:,0:-1] - mean_intensity

        print "Mean duration after normalization:  ", np.mean(X[:,-1])
        print "Mean intensity after normalization: ", np.mean(X[:,0:-1])

        return X, Y

    # Called by main
    # partition the data into training, testing, and validation sets
    def partition_for_training(self, X, Y, trprop, teprop):
        trX = np.zeros((0,X.shape[1]), dtype=np.float32)
        teX = np.zeros((0,X.shape[1]), dtype=np.float32)
        vaX = np.zeros((0,X.shape[1]), dtype=np.float32)

        trY = np.zeros((0,2), dtype=np.float32)
        teY = np.zeros((0,2), dtype=np.float32)
        vaY = np.zeros((0,2), dtype=np.float32)

        for i in range(0,X.shape[0]):
            r = random.random()
            if r < trprop:
                trX = np.concatenate((trX, np.reshape(X[i,:], (1,-1))), axis=0)
                trY = np.concatenate((trY, np.reshape(Y[i,:], (1,2))), axis=0)
            elif r < (trprop + teprop):
                teX = np.concatenate((teX, np.reshape(X[i,:], (1,-1))), axis=0)
                teY = np.concatenate((teY, np.reshape(Y[i,:], (1,2))), axis=0)
            else:
                vaX = np.concatenate((vaX, np.reshape(X[i,:], (1,-1))), axis=0)
                vaY = np.concatenate((vaY, np.reshape(Y[i,:], (1,2))), axis=0)

        return trX, trY, teX, teY, vaX, vaY

    # Called by main
    # save the data to disk
    def store_training_partitions(self, trX, trY, teX, teY, vaX, vaY, location):
        flocation = os.path.join(os.getcwd(), location)
        trlocation = os.path.join(flocation, "training")
        valocation = os.path.join(flocation, "validation")
        telocation = os.path.join(flocation, "testing")
        if (os.path.exists(flocation)):
            print "Location:", location, "exists. Overwriting..."
            if not (os.path.exists(trlocation)):
                os.makedirs(trlocation)
            if not (os.path.exists(valocation)):
                os.makedirs(valocation)
            if not (os.path.exists(telocation)):
                os.makedirs(telocation)

        else:
            print "Location:", location, "does not exist. Creating..."
            os.makedirs(flocation)
            os.makedirs(trlocation)
            os.makedirs(valocation)
            os.makedirs(telocation)

        print

        np.save(os.path.join(trlocation,'trX.npy'), trX)
        np.save(os.path.join(trlocation,'trY.npy'), trY)
        np.save(os.path.join(telocation,'teX.npy'), teX)
        np.save(os.path.join(telocation,'teY.npy'), teY)
        np.save(os.path.join(valocation,'vaX.npy'), vaX)
        np.save(os.path.join(valocation,'vaY.npy'), vaY)

    # Called by main
    # Ramdomly partition data into ~equal size folds
    def partition_for_cross_validation(self, X, Y, folds):
        prop = 1.0/folds
        ret = [[np.zeros((0,X.shape[1]), dtype=np.float32), np.zeros((0,2),
                dtype=np.float32)] for i in range(0,folds)]

        for i in range(0,X.shape[0]):
            r = random.random()
            for j in range(0, folds):
                if ((r >= j*prop) and (r < (j+1)*prop)):
                    tmpX = np.concatenate((ret[j][0], np.reshape(X[i,:], (1,-1))), axis=0)
                    tmpY = np.concatenate((ret[j][1], np.reshape(Y[i,:], (1, 2))), axis=0)
                    ret[j] = [tmpX, tmpY]

        return ret

    # Called by main
    # Store each fold to disk in specified location
    def store_cv_folds(self, lst, location):
        flocation = os.path.join(os.getcwd(), location)
        if (os.path.exists(flocation)):
            print "Location:", location, "exists. Overwriting..."
        else:
            print "Location:", location, "does not exist. Creating..."
            os.makedirs(flocation)
        for i in range(0,len(lst)):
            tmpX = lst[i][0]
            tmpY = lst[i][1]
            np.save(os.path.join(flocation, str(i) + 'X.npy'), tmpX)
            np.save(os.path.join(flocation, str(i) + 'Y.npy'), tmpY)

        print
