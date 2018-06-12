from __future__ import division
import numpy as np
import nibabel as nib
import os
from collections import OrderedDict
import sys

# Default batch size (deprecated)
DEF_BATCH_SIZE = 20

class MACAdapter(object):

    def __init__(self, input_dir, dim, folds=None):
        # store dimensionality
        self.dim = dim
        self.folds = folds

        # Set directory locations for all os interactions
        input_dir = os.path.abspath(input_dir)

        if (self.folds == None):
            self.train_dir = os.path.join(input_dir, "training")
            self.val_dir = os.path.join(input_dir, "validation")
            self.test_dir = os.path.join(input_dir, "testing")

            # Check to make sure input directory is valid
            self._check_io()

            # Load entire dataset into memory (it's plenty small)
            self.trX = np.load(os.path.join(self.train_dir, 'trX.npy'))
            self.trL = np.load(os.path.join(self.train_dir, 'trY.npy'))
            self.trY = np.stack((self.trL[:,1],np.subtract(1,self.trL[:,1])), axis=1)
            self.teX = np.load(os.path.join(self.test_dir, 'teX.npy'))
            self.teL = np.load(os.path.join(self.test_dir, 'teY.npy'))
            self.teY = np.stack((self.teL[:,1],np.subtract(1,self.teL[:,1])), axis=1)
            self.vaX = np.load(os.path.join(self.val_dir, 'vaX.npy'))
            self.vaL = np.load(os.path.join(self.val_dir, 'vaY.npy'))
            self.vaY = np.stack((self.vaL[:,1],np.subtract(1,self.vaL[:,1])), axis=1)

            # get and store sizes
            self.tr_size = self.trX.shape[0]
            self.va_size = self.vaX.shape[0]
            self.te_size = self.teX.shape[0]

            self.second_dim = self.trX.shape[1]

        else:
            self.lst = [[np.load(os.path.join(input_dir,str(fold) + "X.npy")),
                         np.load(os.path.join(input_dir,str(fold) + "Y.npy"))]
                         for fold in range(0,self.folds)]

            self.second_dim = self.lst[0][0].shape[1]

            self.fold = None
            self.trX = None
            self.trY = None
            self.vaX = None
            self.vaY = None


    def _check_io(self):
        # Ensure training directory has the correct structure
        if not os.path.exists(self.train_dir):
            print "Training directory: %s does not exist - ERROR"%self.train_dir
            sys.exit(0)
        if not os.path.exists(self.val_dir):
            print "Validation directory: %s does not exist - ERROR"%self.val_dir
            sys.exit(0)
        if not os.path.exists(self.test_dir):
            print "Testing directory: %s does not exist - ERROR"%self.test_dir
            sys.exit(0)

    def set_fold(self, fold):
        self.fold = fold

        self.trX = np.zeros((0, self.second_dim))
        self.trL = np.zeros((0,2))
        self.teX = np.zeros((0, self.second_dim))
        self.teY = np.zeros((0,2))
        self.vaX = np.zeros((0, self.second_dim))
        self.vaL = np.zeros((0,2))

        for i in range(0, self.folds):
            if i == fold:
                self.vaX = np.concatenate((self.vaX, self.lst[i][0]), axis=0)
                self.vaL = np.concatenate((self.vaL, self.lst[i][1]), axis=0)
            else:
                self.trX = np.concatenate((self.trX, self.lst[i][0]), axis=0)
                self.trL = np.concatenate((self.trL, self.lst[i][1]), axis=0)

        self.trY = np.stack((self.trL[:,1],np.subtract(1,self.trL[:,1])), axis=1)
        self.vaY = np.stack((self.vaL[:,1],np.subtract(1,self.vaL[:,1])), axis=1)


        self.tr_size = self.trX.shape[0]
        self.va_size = self.vaX.shape[0]
        self.te_size = 0

    def _get_rand_array(self, length, mx):
        return np.random.randint(0, high=mx, size=length)

    # This is real slow for large batches...
    def get_batch(self, size=DEF_BATCH_SIZE, collection = None):
        if collection == None or collection == "training":
            size = min(size, self.tr_size)
            arr = self._get_rand_array(size, self.tr_size)
            X = np.take(self.trX, arr, axis=0).reshape([size, self.second_dim, 1])
            Y = np.take(self.trY, arr, axis=0)
        if collection == "validation":
            # validation always returns full validation set
            X = self.vaX.reshape([self.va_size, self.second_dim, 1])
            Y = self.vaY
        if collection == "testing":
            # testing always returns the full testing set
            X = self.teX.reshape([self.te_size, self.second_dim, 1])
            Y = self.teY

        durs = X[:,-1]
        x = np.reshape(X[:,0:-1], [-1,37,self.dim,1])

        return x, durs, Y

    def get_ids(self):
        return self.trL[:,0], self.vaL[:,0], self.teL[:,0]
