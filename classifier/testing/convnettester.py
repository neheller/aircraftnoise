import tensorflow as tf
import sys
import os
import numpy as np

'''
The object that encapsulates the training procedure and status
'''
class ConvNetTester(object):

    '''
    The object constructor
    '''
    # net:             the model object we are training
    # server:          the server object to use for all operating system
    #                  interactions
    # trainer:         the object which trains the model
    def __init__(self, net, server, trainer):
        # keep pre-constructed objects
        self.net = net
        self.server = server
        self.trainer = trainer

    def show_progress(self, accuracy, precision, recall, f1,
                        folds, tries, tri, cv_accuracy_sum, cv_precision_sum,
                        cv_recall_sum, cv_f1_sum, fold_accuracy_sum, fold_precision_sum,
                        fold_recall_sum, fold_f1_sum, fold):
        self.server.log("****************************************", "warning")
        progress = 1.0*(fold+1)/folds + 1.0*(tri+1)/tries*(1.0/folds)
        self.server.log("", "warning")
        self.server.log("Progress: " + str(progress), "warning")
        self.server.log("Fold: " + str(fold+1) + " of " + str(folds), "warning")
        self.server.log("Trial: " + str(tri) + " of " + str(tries), "warning")
        tri_accuracy = fold_accuracy_sum/max(tri,1)
        cv_accuracy = cv_accuracy_sum/max(fold,1)
        self.server.log("", "warning")
        self.server.log("This Accuracy: " + str(accuracy), "warning")
        self.server.log("Fold Accuracy: " + str(tri_accuracy), "warning")
        self.server.log("Overall Accuracy: " + str(cv_accuracy), "warning")
        tri_precision = fold_precision_sum/max(tri,1)
        cv_precision = cv_precision_sum/max(fold,1)
        self.server.log("", "warning")
        self.server.log("This Precision: " + str(precision), "warning")
        self.server.log("Fold Precision: " + str(tri_precision), "warning")
        self.server.log("Overall Precision: " + str(cv_precision), "warning")
        tri_recall = fold_recall_sum/max(tri,1)
        cv_recall = cv_recall_sum/max(fold,1)
        self.server.log("", "warning")
        self.server.log("This Recall: " + str(recall), "warning")
        self.server.log("Fold Recall: " + str(tri_recall), "warning")
        self.server.log("Overall Recall: " + str(cv_recall), "warning")
        tri_f1 = fold_f1_sum/max(tri,1)
        cv_f1 = cv_f1_sum/max(fold,1)
        self.server.log("", "warning")
        self.server.log("This F1: " + str(f1), "warning")
        self.server.log("Fold F1: " + str(tri_f1), "warning")
        self.server.log("Overall F1: " + str(cv_f1), "warning")
        self.server.log("", "warning")
        self.server.log("****************************************", "warning")



    def run_cross_validation(self, folds, trials_per_fold):
        cv_accuracy_sum = 0
        cv_precision_sum = 0
        cv_recall_sum = 0
        cv_f1_sum = 0

        for fold in range(0, folds):
            # Set adapter to correct train/test sets
            self.server.adapter.set_fold(fold)
            # Initialize accumulators to zero

            fold_accuracy_sum = 0
            fold_precision_sum = 0
            fold_recall_sum = 0
            fold_f1_sum = 0
            tri = 0
            while tri < trials_per_fold:
                accuracy, precision, recall, f1 = self.trainer.train()
                if not np.isnan(f1):
                    tri = tri + 1
                    fold_accuracy_sum = fold_accuracy_sum + accuracy
                    fold_precision_sum = fold_precision_sum + precision
                    fold_recall_sum = fold_recall_sum + recall
                    fold_f1_sum = fold_f1_sum + f1
                else:
                    accuracy = 0
                    self.server.log("DUD", "warning")
                self.show_progress(accuracy, precision, recall, f1,
                                    folds, trials_per_fold, tri, cv_accuracy_sum,
                                    cv_precision_sum, cv_recall_sum, cv_f1_sum,
                                    fold_accuracy_sum, fold_precision_sum,
                                    fold_recall_sum, fold_f1_sum, fold)

            # Accumulate over full cross validation
            cv_accuracy_sum = cv_accuracy_sum + fold_accuracy_sum/trials_per_fold
            cv_precision_sum = cv_precision_sum + fold_precision_sum/trials_per_fold
            cv_recall_sum = cv_recall_sum + fold_recall_sum/trials_per_fold
            cv_f1_sum = cv_f1_sum + fold_f1_sum/trials_per_fold
