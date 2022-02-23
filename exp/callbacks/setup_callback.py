import os, uuid
from exp.callbacks.core import Callback


class SetupCallback(Callback):
    """
    Performs setup procedures to setup an experiment (i.e. create local directory to store
    artifacts etc.) as well as updating training progress between batches and epochs.
    """
    def __init__(self, display_every_n_batch=25, eval_every_n_epoch=1, save_every_n_epoch=1,
                 save_dir='runs'):
        super(SetupCallback, self).__init__()
        self.display_every_n_batch = display_every_n_batch
        self.eval_every_n_epoch = eval_every_n_epoch
        self.save_every_n_epoch = save_every_n_epoch
        self.save_dir = save_dir
        
        
    def before_train(self):
        # setup exp directory
        if not hasattr(self.exp,'_id'):
            self.exp._id = f'exp_{uuid.uuid4()}'
        self.exp.save_dir = f'./{self.save_dir}/{self.exp._id}'
        os.makedirs(self.exp.save_dir, exist_ok=True)
        
        # init progress
        if not hasattr(self.exp, 'n_epochs'): self.exp.n_epochs=0.
        if not hasattr(self.exp, 'n_iterations') : self.exp.n_iter=0
        
        # set display/eval frequencies
        self.exp.display_every_n_batch = self.display_every_n_batch
        self.exp.eval_every_n_epoch = self.eval_every_n_epoch
        self.exp.save_every_n_epoch = self.save_every_n_epoch
        
        # set to training mode
        self.exp.in_train = True
        
        # reset stop to False
        self.exp.stop = False
        
        # gather params to learn
        self.exp.weights, self.exp.biases = list(), list()
        for param_name, param in self.model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'): self.exp.biases.append(param)
                else: self.exp.weights.append(param)
                    
        # init custom_loss flag to False to indicate typical loss compute pattern
        self.exp.custom_loss = False
        
        # set setup flag to indicate procedure has been ran
        self.exp.initialized = True
        
        
    def before_epoch(self):
        self.model.train()
        self.exp.in_train=True
    
    
    def after_batch(self):
        self.exp.n_iter+=1
    
    
    def after_epoch(self):
        self.exp.n_epochs+=1
        self.exp.n_iter = 0  # reset batch count for the new epoch
        # save every n epochs
        if self.exp.n_epochs % self.save_every_n_epoch == 0:
            if hasattr(self.exp, 'opt'):
                self.exp.save(self.exp.opt)
            else:
                self.exp.save()
        # eval every n epochs
        if self.exp.n_epochs % self.eval_every_n_epoch == 0:
            # evaluate on training set
            if self.exp.data.train_dl:
                self.exp.evaluate(self.exp.data.train_dl, train=True)
            # evaluate on validation set
            if self.exp.data.valid_dl:
                self.exp.evaluate(self.exp.data.valid_dl)
    
    
    def before_train_eval(self):
        ...  # placeholder
        self.before_eval()
    
    
    def after_train_eval(self):
        ... # placeholder
        self.after_eval()
        
    
    def before_eval(self):
        self.model.eval()
        self.exp.in_train=False
        
        
    def after_eval(self):
        self.model.train()
        self.exp.in_train=True