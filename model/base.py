from collections import OrderedDict
from torch import nn
from torchvision.models import vgg16
from model.utils import add_module, decimate


class VGGBase(nn.Module):
    """
    Architecture base for the overall detector model
    """
    def __init__(self):
        super(VGGBase, self).__init__()
        
        # define module components
        self.conv1 = add_module({
            # convs1
            'conv1_1' : nn.Conv2d(3, 64, kernel_size=3, padding=1),
            'relu1_1' : nn.ReLU(inplace=True),
            'conv1_2' : nn.Conv2d(64, 64, kernel_size=3, padding=1),
            'relu1_2' : nn.ReLU(inplace=True),
            'ftmap1'  : nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        })
        
        self.conv2 = add_module({            
            # convs2
            'conv2_1' : nn.Conv2d(64, 128, kernel_size=3, padding=1),
            'relu2_1' : nn.ReLU(inplace=True),
            'conv2_2' : nn.Conv2d(128, 128, kernel_size=3, padding=1),
            'relu2_2' : nn.ReLU(inplace=True),
            'ftmap2'  : nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        })
        
        self.conv3 = add_module({
            # convs3
            'conv3_1' : nn.Conv2d(128, 256, kernel_size=3, padding=1),
            'relu3_1' : nn.ReLU(inplace=True),
            'conv3_2' : nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'relu3_2' : nn.ReLU(inplace=True),
            'conv3_3' : nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'relu3_3' : nn.ReLU(inplace=True),
            'ftmap3'  : nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        })

        self.conv4 = add_module({            
            # convs4
            'conv4_1' : nn.Conv2d(256, 512, kernel_size=3, padding=1),
            'relu4_1' : nn.ReLU(inplace=True),
            'conv4_2' : nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'relu4_2' : nn.ReLU(inplace=True),
            'conv4_3' : nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'relu4_3' : nn.ReLU(inplace=True),
            'ftmap4'  : nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        })
        
        self.conv5 = add_module({
            # convs5
            'conv5_1' : nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'relu5_1' : nn.ReLU(inplace=True),
            'conv5_2' : nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'relu5_2' : nn.ReLU(inplace=True),
            'conv5_3' : nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'relu5_3' : nn.ReLU(inplace=True),
            'ftmap5'  : nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        })
        
        self.avgPool  = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.conv6 = add_module({
            # convs6
            'conv6'  : nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),  # atrous convolution
            'ftmap6' : nn.ReLU(inplace=True)
        })
        
        self.conv7 = add_module({
            'conv7'  : nn.Conv2d(1024, 1024, kernel_size=1),
            'ftmap7' : nn.ReLU(inplace=True)
        })

        # load pre-trained parameters from vgg16
        self.load_params()


    def transfer_params(self, params, trained_params):            
        names  = list(params.keys())
        _names = list(trained_params.keys())
        assert len(names) == len(_names), "parameter length mis-alignment"
        # transfer weights/biases from pretrained VGG16.features module
        for i, n in enumerate([p for p in names]):
            # get corresponding name in the pretrained model
            _n = _names[i]
            assert params[n].shape == trained_params[_n].shape, "param shape mis-alignment"                        
            params[n] = trained_params[_n].detach().clone()


    def load_params(self):
        # get pretrained vgg16 params
        vgg = vgg16(pretrained=True)
        _params = vgg.features.state_dict()
        
        # load pre-trained params for conv layers 1-5
        modules = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        params  = OrderedDict()
        for module in modules:
            for k, v in module.state_dict().items():
                params[k] = v
        # copy over parameters & load state dict
        self.transfer_params(params, _params)
        for module in modules:
            # get subset of params in module            
            params_subset = {k: v for k, v in params.items() if k in module.state_dict().keys()}
            module.load_state_dict(params_subset)
        
        # load pre-trained params for layers 6-7
        w6 = vgg.classifier[0].weight.detach().clone().view(4096,512,7,7)
        b6 = vgg.classifier[0].bias.detach().clone()
        w7 = vgg.classifier[3].weight.detach().clone().view(4096,4096,1,1)
        b7 = vgg.classifier[3].bias.detach().clone()
        self.conv6.load_state_dict({
            'conv6.weight': decimate(w6, m=[4, None, 3, 3]),
            'conv6.bias'  : decimate(b7, m=[4])
        })
        self.conv7.load_state_dict({
            'conv7.weight': decimate(w7, m=[4, 4, 3, 3]),
            'conv7.bias'  : decimate(b7, m=[4])
        })

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        ftmap4 = self.conv4(x)
        x = self.conv5(ftmap4)
        x = self.avgPool(x)
        x = self.conv6(x)
        ftmap7 = self.conv7(x)
        return ftmap4, ftmap7