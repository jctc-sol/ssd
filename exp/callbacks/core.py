import re, uuid
from exp.utils.core import listify, camel2snake


class Callback():
    """
    Base template for all other callback functions to inhereit from.
    """
    # _order filed will be used to sort order of callback to execute;
    # 0 being the base layer parent Callback class
    _order=0

    def set_exp(self, exp): 
        self.exp=exp
        
        
    def rmv_exp(self):
        if hasattr(self, 'exp'): delattr(self, 'exp')


    def find_differential_lr(self):
        # used for callbacks that need to adjust LR
        # find the relative ratio of LR for all param groups wrt to the first 
        # param group; this is for differential learning scenarios where some 
        # params groups are updated at higher LR than others
        self.lr = listify(self.exp.opt.param_groups)[0]['lr']
        self.lr_ratios = [1.0]
        for pg in listify(self.exp.opt.param_groups):
            self.lr_ratios.append(pg['lr']/self.lr)
        

    def set_lr(self, lr):
        # used for callbacks that need to adjust LR
        # set current lr value; update all optimizer param group
        # learning rates to new value * their respective lr ratio
        self.lr = lr
        for i, pg in enumerate(listify(self.exp.opt.param_groups)):
            pg['lr'] = lr * self.lr_ratios[i]
            
            
    def __getattr__(self, k): 
        return getattr(self.exp, k)  # reaches into experiment to get attribute k
    
    
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')
    
    
    def __call__(self, cb_name):  # check if the callback specified by cb_name exists        
        f = getattr(self, cb_name, None)
        # below condition set to not f() to avoid writing "return True" in every callback function
        if f and not f(): return True
        return False


