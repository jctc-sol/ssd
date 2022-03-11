import numpy as np
from exp.utils.core import camel2snake


class Recorder(object):
    """
    Keeps running record on max/min/avg/sum and cnt
    """
    fields = ['val', 'sum', 'cnt', 'min', 'max', 'avg', 'exp_avg']

    def __init__(self, name='', beta=0.3):
        self.reset()
        self.name = camel2snake(name + 'Recorder')
        self.beta = beta
        
        
    def reset(self):
        for f in self.fields: setattr(self, f, 0)
        self.value, self.weight, self.smpAvg, self.expAvg = [], [], [], []
        
        
    def update(self, val, wgt=1):
        # check for infinite input value
        if val == float("inf"): val = 1e12
        elif val == -1*float("inf"): val = -1e12
        # check for nan input values
        if val == np.nan: return
        # update running stats
        self.val = val
        self.sum += val*wgt
        self.cnt += wgt
        self.avg = self.sum/self.cnt
        self.exp_avg = self.beta*self.exp_avg + (1-self.beta)*val*wgt
        # update min/max values
        if (val < self.min): self.min = val
        if (val > self.max): self.max = val
        # record to tapes
        self.value.append(val)
        self.weight.append(wgt)
        self.smpAvg.append(self.avg)
        self.expAvg.append(self.exp_avg)
        
        
    def __call__(self, val, n=1):
        self.update(val, n)
        
        
    def __repr__(self):
        return self.name
