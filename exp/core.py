import torch
from exp.utils.core import listify
from exp.callbacks import *


class Exp():
    
    def __init__(self, learner, cbs=None, cb_funcs=None, device=None, name=None, desc=None):
        # information about the experiment
        self.name = name
        self.desc = desc
        # learner object; wrapper about dataset, model, loss function, and metrics
        self.learner = learner
        # initialization
        self.stop, self.cbs = False, []
        
        # list of instantiate callback rountines
        cbs = listify(cbs)
        # iterate over list of callback functions to be initialized and added as callback rountine
        for cbf in listify(cb_funcs):
            cb = cbf()      # instantiate each callback function
            cbs.append(cb)  # append callback function to list
        for cb in cbs:
            self.add_callback(cb)
        
        # setup device
        if device:
            if type(device)==torch.device: 
                self.device=device
            elif type(device)==str:
                self.device=torch.device(device)
        else:
            # defaults to cuda:0 if cuda is available
            if torch.cuda.is_available(): self.device = torch.device('cuda:0') 
            else: self.device = torch.device('cpu')

        # move model & cost function to device
        self.model.to(self.device)
        self.loss_func.to(self.device)

        # call before_train rountine to perform the rest of experiment setup
        self('before_train')
        
        
    def __call__(self, cb_name):
        # default to false
        res = False
        # sort the order of callbacks to execute 
        for cb in sorted(self.cbs, key=lambda x: x._order): 
            res = cb(cb_name) or res
        return res

    
    def __repr__(self):
        if hasattr(self,'_id'): return f'exp_{self._id}\n{self.name}; {self.desc}'
        else: return f'exp\n{self.name}; {self.desc}'
    

    def add_callback(self, cb):
        # if cb is a list of callbacks; recursively add each one
        if isinstance(cb, list):
            for _cb in cb: self.add_callback(_cb)
        else:
            # avoid adding duplicate callbacks by checking for unique names        
            for _cb in self.cbs:
                if not cb or cb.name == _cb.name: return
            self.cbs.append(cb)
            # add callback name to current Exp name-space
            setattr(self, cb.name, cb)
            # link callback to this experiment
            cb.set_exp(self)
        
    
    def rmv_callback(self, cb):
        # if cb is a list of callbacks; recursively remove each one
        if isinstance(cb, list):
            for _cb in cb: self.rmv_callback(cb)
        else:
            cbs = []
            for _cb in self.cbs:
                if cb.name != _cb.name: cbs.append(_cb)
            self.cbs = cbs
            # remove callback name from current Exp name-space
            delattr(self, cb.name)
            # unlink callback from this experiment
            cb.rmv_exp()
    
    
    # pass-along property methods that makes it easier to access learner properties from runner object
    @property
    def model(self):     return self.learner.model
    @property
    def loss_func(self): return self.learner.loss_func
    @property
    def data(self):      return self.learner.data    
    @property
    def metrics(self):   return self.learner.metrics
