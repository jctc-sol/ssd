import numpy as np
import torch, torchvision
from torch.utils.tensorboard import SummaryWriter
from exp.callbacks.core import Callback


class TensorBoardCallback(Callback):

    def __init__(self, log_per_n_steps=10):
        Callback.__init__(self)
        self.n = log_per_n_steps
        self.writer = SummaryWriter()
        self._order = 1  # set to run after callbacks with _order == 0
    

    def before_train(self):
        # obtain a sample batch of input image
        train_batch = next(iter(self.exp.data.train_dl))        
        valid_batch = next(iter(self.exp.data.valid_dl))
        img_train = train_batch['images'].to(self.exp.device)
        img_valid = valid_batch['images'].to(self.exp.device)   
        # make image grid & add images to writer
        train_grid = torchvision.utils.make_grid(img_train)
        self.writer.add_image('train_images', train_grid, 0)
        valid_grid = torchvision.utils.make_grid(img_valid)
        self.writer.add_image('valid_images', valid_grid, 0)
        # add model as graph to writer
        self.writer.add_graph(model=self.model, 
                              input_to_model=img_train)        
    

    def after_loss(self):
        if self.exp.n_iter % self.n == 0:
            # record loss as scalar
            if self.exp.in_train:
                self.writer.add_scalar('train_loss', self.exp.loss.item(), self.exp.n_iter)
            else:
                self.writer.add_scalar('valid_loss', self.exp.loss.item(), self.exp.n_iter)


    def after_backward(self):
        self.log_params(grad=True)


    def after_step(self):
        self.log_params()


    def log_params(self, grad=False):
        if self.exp.n_iter % self.n == 0:
            for tag, param in self.exp.model.named_parameters():
                if grad and param.grad is not None: 
                    tag = tag + '.grad'
                    x = param.grad.data.cpu().numpy()
                else:
                    x = param.data.cpu().numpy()
                if self.nan_or_infinite(x):
                    print(f"WARNING: global step {self.exp.n_iter}; {tag} contains nan or infinite!")
                else:
                    if self.all_zeros(x) and 'bias' not in tag:
                        print(f"WARNING: global step {self.exp.n_iter}; {tag} is all zeros!")
                    self.writer.add_histogram(tag, x, self.exp.n_iter)


    def nan_or_infinite(self, x):
        # check to see if any entry in x is np.nan or np.infinite
        if np.sum(~np.isfinite(x)) > 0:
            return True
        else:
            return False


    def all_zeros(self, x):
        # check to see if all entry in x is 0
        if np.sum(np.abs(x)) == 0:
            return True
        else:
            return False


    def close_writer(self):
        self.writer.flush()
        self.writer.close()


    def after_train(self):
        self.close_writer()


    def after_cancel_epoch(self):
        self.close_writer()


    def after_cancel_train(self):
        self.close_writer()