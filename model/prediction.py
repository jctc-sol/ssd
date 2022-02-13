import torch
from torch import nn
from model.utils import add_module, he_init


def permute_reshape(x, bs, n):
    # note: contiguous() ensures tensor is stored in a contiguous chunk of memory; 
    # needed for calling .view() for reshaping below
    x = x.permute(0,2,3,1).contiguous()
    x = x.view(bs, -1, n)
    return x


class Locators(nn.Module):
    """
    Prediction convolution layers to produce location predictions in the 
    form of offset coordinates wrt to prior bounding boxes
    """
    def __init__(self, n_boxes):
        super(Locators, self).__init__()
        
        # localizations
        self.loc4  = nn.Conv2d(512, n_boxes['ftmap4']*4, kernel_size=3, padding=1)
        self.loc7  = nn.Conv2d(1024, n_boxes['ftmap7']*4, kernel_size=3, padding=1)
        self.loc8  = nn.Conv2d(512, n_boxes['ftmap8']*4, kernel_size=3, padding=1)
        self.loc9  = nn.Conv2d(256, n_boxes['ftmap9']*4, kernel_size=3, padding=1)
        self.loc10 = nn.Conv2d(256, n_boxes['ftmap10']*4, kernel_size=3, padding=1)
        self.loc11 = nn.Conv2d(256, n_boxes['ftmap11']*4, kernel_size=3, padding=1)
    
    
    def forward(self, ftmap4, ftmap7, ftmap8, ftmap9, ftmap10, ftmap11):
        bs = ftmap4.size(0)     # get batch size        
        x = self.loc4(ftmap4)                 # (N, 16, 38, 38)
        loc4 = permute_reshape(x, bs, 4)      # (N, 5776, 4), for a total of 5776 prior bounding boxes
        x = self.loc7(ftmap7)                 # (N, 24, 19, 19)
        loc7 = permute_reshape(x, bs, 4)      # (N, 2166, 4)
        x = self.loc8(ftmap8)                 # (N, 24, 19, 19)
        loc8 = permute_reshape(x, bs, 4)      # (N, 600, 4)
        x = self.loc9(ftmap9)                 # (N, 24, 5, 5)
        loc9 = permute_reshape(x, bs, 4)      # (N, 150, 4)
        x = self.loc10(ftmap10)               # (N, 16, 3, 3)
        loc10 = permute_reshape(x, bs, 4)     # (N, 36, 4)
        x = self.loc11(ftmap11)               # (N, 16, 1, 1)
        loc11 = permute_reshape(x, bs, 4)     # (N, 4, 4)

        return loc4, loc7, loc8, loc9, loc10, loc11


class Classifiers(nn.Module):
    """
    Prediction convolution layers to produce object class probabilities
    """
    def __init__(self, n_boxes, n_cls):
        super(Classifiers, self).__init__()
        self.n_cls = n_cls # get # of class categories
        
        # classifications
        self.cls4  = nn.Conv2d(512, n_boxes['ftmap4']*n_cls, kernel_size=3, padding=1)
        self.cls7  = nn.Conv2d(1024, n_boxes['ftmap7']*n_cls, kernel_size=3, padding=1)
        self.cls8  = nn.Conv2d(512, n_boxes['ftmap8']*n_cls, kernel_size=3, padding=1)
        self.cls9  = nn.Conv2d(256, n_boxes['ftmap9']*n_cls, kernel_size=3, padding=1)
        self.cls10 = nn.Conv2d(256, n_boxes['ftmap10']*n_cls, kernel_size=3, padding=1)
        self.cls11 = nn.Conv2d(256, n_boxes['ftmap11']*n_cls, kernel_size=3, padding=1)
    
    
    def forward(self, ftmap4, ftmap7, ftmap8, ftmap9, ftmap10, ftmap11):                
        bs = ftmap4.size(0) # get batch size
        x = self.cls4(ftmap4)                      # (N, 4 boxes * n_classes, 38, 38)
        cls4 = permute_reshape(x, bs, self.n_cls)  # (N, 5776, n_classes)
        x = self.cls7(ftmap7)                      # (N, 6 boxes * n_classes, 19, 19)
        cls7 = permute_reshape(x, bs, self.n_cls)  # (N, 2166, n_classes)
        x = self.cls8(ftmap8)                      # (N, 6 boxes * n_classes, 10, 10)
        cls8 = permute_reshape(x, bs, self.n_cls)  # (N, 600, n_classes)
        x = self.cls9(ftmap9)                      # (N, 6 boxes * n_classes, 5, 5)
        cls9 = permute_reshape(x, bs, self.n_cls)  # (N, 150, n_classes)
        x = self.cls10(ftmap10)                    # (N, 4 boxes * n_classes, 3, 3)
        cls10 = permute_reshape(x, bs, self.n_cls) # (N, 36, n_classes)
        x = self.cls11(ftmap11)                    # (N, 4 boxes * n_classes, 1, 1)
        cls11 = permute_reshape(x, bs, self.n_cls) # (N, 4, n_classes)
        return cls4, cls7, cls8, cls9, cls10, cls11
    

class PredLayers(nn.Module):
    """
    Prediction conv layers to produce:
    a) location predictions in the form of offset coordinates wrt to prior bounding boxes; and 
    b) object class probabilities
    """
    def __init__(self, n_classes):
        super(PredLayers, self).__init__()
        
        # Define how many bounding boxes (i.e. each with a different aspect ratio)
        # there to be per ftmap
        n_boxes = {'ftmap4' : 4,
                   'ftmap7' : 6,
                   'ftmap8' : 6,
                   'ftmap9' : 6,
                   'ftmap10': 4,
                   'ftmap11': 4}
        
        # create locators and classifiers
        self.locators = Locators(n_boxes)
        self.classifiers = Classifiers(n_boxes, n_classes)
        
        # Initalize all convolution parameters
        kaiming_params = {
            'a': 0,
            'mode': 'fan_in',
            'nonlinearity': 'relu',
        }
        he_init(self.locators.children(), **kaiming_params)
        he_init(self.classifiers.children(), **kaiming_params)

        
    def forward(self, ftmap4, ftmap7, ftmap8, ftmap9, ftmap10, ftmap11):        
        # localizations
        loc4, loc7, loc8, loc9, loc10, loc11 = self.locators(ftmap4, ftmap7, ftmap8, ftmap9, ftmap10, ftmap11)
        # classifications
        cls4, cls7, cls8, cls9, cls10, cls11 = self.classifiers(ftmap4, ftmap7, ftmap8, ftmap9, ftmap10, ftmap11)
        # Concatenate all locators and all classifiers
        # There are a total of 5776 + 2166 + 600 + 150 + 36 + 4 = 8732 bounding box locations in total
        locations  = torch.cat([loc4, loc7, loc8, loc9, loc10, loc11], dim=1)
        cls_scores = torch.cat([cls4, cls7, cls8, cls9, cls10, cls11], dim=1)
        return locations, cls_scores