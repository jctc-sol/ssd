import os, sys, time, uuid
import torch.nn.functional as F
import torchvision.transforms.functional as FT
from functools import partial
from torch import nn
from datasets import coco
from utils   import *
from model   import *
from metric  import *
from loss    import *
from torch_exp.learner import Learner
DEVICE = 'cpu' if torch.cuda.device_count() == 0 else 'cuda:0'
print(DEVICE)