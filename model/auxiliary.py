from torch import nn
from model.utils import add_module, he_init


class AuxLayers(nn.Module):
    """
    Auxiliary layers subsequent to the VGG base module
    """
    def __init__(self):
        super(AuxLayers, self).__init__()
        
        self.conv8 = add_module({
            'conv8_1' : nn.Conv2d(1024, 256, kernel_size=1, padding=0),
            'relu8_1' : nn.ReLU(inplace=True),
            'conv8_2' : nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            'ftmap8'  : nn.ReLU(inplace=True)            
        })
        self.conv9 = add_module({
            'conv9_1' : nn.Conv2d(512, 128, kernel_size=1, padding=0),
            'relu9_1' : nn.ReLU(inplace=True),
            'conv9_2' : nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            'ftmap9'  : nn.ReLU(inplace=True)
        })
        self.conv10 = add_module({
            'conv10_1': nn.Conv2d(256, 128, kernel_size=1, padding=0),
            'relu10_1': nn.ReLU(inplace=True),
            'conv10_2': nn.Conv2d(128, 256, kernel_size=3, padding=1),
            'ftmap10' : nn.ReLU(inplace=True)
        })
        self.conv11 = add_module({
            'conv11_1': nn.Conv2d(256, 128, kernel_size=1, padding=0),
            'relu11_1': nn.ReLU(inplace=True),
            'conv11_2': nn.Conv2d(128, 256, kernel_size=3, padding=1),
            'ftmap11' : nn.ReLU(inplace=True)
        })    
        # init layer parameters
        kaiming_params = {
            'a': 0,
            'mode': 'fan_in',
            'nonlinearity': 'relu',
        }
        he_init(self.children(), **kaiming_params)
                        
                
    def forward(self, x):
        ftmap8 = self.conv8(x)        
        ftmap9 = self.conv9(ftmap8)
        ftmap10 = self.conv10(ftmap9)
        ftmap11 = self.conv11(ftmap10)
        # return feature maps
        return ftmap8, ftmap9, ftmap10, ftmap11
