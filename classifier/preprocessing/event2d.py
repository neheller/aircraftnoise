import tensorflow as tf
import numpy as np
import json
import math
import sys

keys = ["6.3","8.0","10.0","12.5","16.0","20.0","25.0","31.5","40.0","50.0","63.0","80.0","100","125","160","200","250","315","400","500","630","800","1000","1250","1600","2000","2500","3150","4000","5000","6300","8000","10000","12500","16000","20000","overall"]


class Event2D:
    def lin_interp(self, rawmat, dim):
        ncol = np.shape(rawmat)[0]
        ret = np.empty([ncol, dim], dtype=float)
        rmx = np.shape(rawmat)[1] - 1
        inc = 1.0*(rmx-1)/dim
        start = inc/2
        for i in range(ncol):
            for j in range(dim):
                rw = start + j*inc
                ind1 = int(math.floor(rw))
                weight1 = 1-(rw-ind1)
                #ind2 = min(ind1 + 1, rmx-1)
                ind2 = ind1+1
                weight2 = 1-weight1
                ret[i,j] = weight1*rawmat[i,ind1] + weight2*rawmat[i,ind2]

        return ret

    def to_mat(self, rawdat, duration):
        if ('meta' in rawdat):
            pre = int(rawdat['meta']['pre'])
        elif ('ehistory' in rawdat):
            pre = int(rawdat['ehistory']['meta']['pre']['length'])
        elif ('hist' in rawdat):
            pre = 0
        rawmat = np.empty([len(keys),duration])
        i = 0
        try:
            rawdat['data']
            for key in keys:
                j = 0
                k = 0
                for t in rawdat['data'][key]:
                    if ((j > pre) and (j < pre + duration)):
                        rawmat[i, k] = t
                        k = k + 1
                    j = j + 1
                i = i + 1
        except KeyError, e:
            for key in keys:
                j = 0
                k = 0
                if (key == 'overall'):
                    for t in rawdat['ehistory'][key]:
                        if ((j > pre) and (j < pre + duration)):
                            rawmat[i, k] = t
                            k = k + 1
                        j = j + 1
                else:
                    for t in rawdat['ehistory']['freq'][key]:
                        if ((j > pre) and (j < pre + duration)):
                            rawmat[i, k] = t
                            k = k + 1
                        j = j + 1
                i = i + 1

        return rawmat

    def get_data(self, rawdat, duration, dim):
        rawmat = self.to_mat(rawdat, duration)
        return self.lin_interp(rawmat, dim)

    def __init__(self, row, dim, src='file'):
        self.dim = dim
        self.flag = 0
        if (src=='file'):
            self.id = int(row[0])
            rawdat = json.loads(row[8])
        else:
            self.id = row['eventid']
            rawdat = row
        if not (('ehistory' in rawdat) or ('meta' in rawdat) or ('hist' in rawdat)):
            self.flag = 1
        else:
            if (src=='file'):
                self.label = int('aircraft' in json.loads(row[9]).keys())
            else:
                self.label = int('aircraft' in rawdat['ehistory']['meta'])

        if ('meta' in rawdat):
            self.duration = int(rawdat['meta']['hist'])
        elif ('ehistory' in rawdat):
            self.duration = int(rawdat['ehistory']['meta']['event']['length'])
        elif ('hist' in rawdat):
            self.duration = int(rawdat['hist'])
        else:
            self.flag = 1
        if not self.flag == 1:
            self.data = self.get_data(rawdat, self.duration, dim)

    def present(self, n):
        print self.id

    def to_array(self):
        return np.append(self.data.flatten(),self.duration).reshape((1,37*self.dim+1))
