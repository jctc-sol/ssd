import torch
import torch.nn as nn
from math import sqrt


def add_module(components):
    """
    convenience function for adding individual components into a sequential module
    params:
    components: dict type specifying the layers inside the module in the intended order
    """
    module = nn.Sequential()
    for k,v in components.items():
        module.add_module(k,v)
    return module


def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.
    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.
    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    
    Code: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())
    return tensor


def he_init(layers, normal=True, **kaiming_params):
    """
    Initialize convolution parameters with Kaiming init
    and initialize all bias parameters as zeros
    """
    if normal: 
        method = nn.init.kaiming_normal_
    else: 
        method = nn.init.kaiming_uniform_
    # go through all layers & initialize parameters one at a time
    for c in layers:
        if isinstance(c, nn.Conv2d):
            method(c.weight, **kaiming_params)
            nn.init.constant_(c.bias, 0.)

            
def create_pboxes():
    """
    Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.
    :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
    """
    # size of the kernels in each respective feature map layer in the prediction
    # network after permuting the convolution output
    fmap_dims = {'conv4_3' : 38,
                 'conv7'   : 19,
                 'conv8_2' : 10,
                 'conv9_2' : 5,
                 'conv10_2': 3,
                 'conv11_2': 1}

    # relative scale of each feature map to the input image
    obj_scales = {'conv4_3' : 0.1,
                  'conv7'   : 0.2,
                  'conv8_2' : 0.375,
                  'conv9_2' : 0.55,
                  'conv10_2': 0.725,
                  'conv11_2': 0.9}

    # different aspect ratio bounding boxes to use at each feature map layer
    aspect_ratios = {'conv4_3' : [1., 2., 0.5],
                     'conv7'   : [1., 2., 3., 0.5, .333],
                     'conv8_2' : [1., 2., 3., 0.5, .333],
                     'conv9_2' : [1., 2., 3., 0.5, .333],
                     'conv10_2': [1., 2., 0.5],
                     'conv11_2': [1., 2., 0.5]}

    fmaps = list(fmap_dims.keys())
    prior_boxes = []

    # iterate through each feature map
    for k, fmap in enumerate(fmaps):

        # go through each grid-location within the convolution kernel (i, j)
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):

                # compute bounding box center coordinates normalized against the size of feature map dimension
                cx = (j + 0.5) / fmap_dims[fmap]
                cy = (i + 0.5) / fmap_dims[fmap]

                # populate bounding boxes of different aspect ratio to prior_boxes list
                for ratio in aspect_ratios[fmap]:
                    # bounding boxes defined in terms [center_x_coord, center_y_coord, center_w_coord, center_h_coord]
                    width  = obj_scales[fmap] * sqrt(ratio)
                    height = obj_scales[fmap] / sqrt(ratio)
                    prior_boxes.append([cx, cy, width, height])
                    # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                    # scale of the current feature map and the scale of the next feature map. This results in a 
                    # additional square bounding box with aspect ratio of 1
                    if ratio == 1.:
                        try:
                            current_scale = obj_scales[fmap]
                            next_scale    = obj_scales[fmaps[k+1]]
                            additional_scale = sqrt(current_scale + next_scale)
                        # For the last feature map, there is no "next" feature map (i.e. index out of bound in fmaps[k+1]) 
                        except IndexError:
                            additional_scale = 1.
                        prior_boxes.append([cx, cy, additional_scale, additional_scale])

    prior_boxes = torch.FloatTensor(prior_boxes)  # shape (8732, 4)
    prior_boxes.clamp_(0, 1) # truncate all values between [0,1]

    return prior_boxes    