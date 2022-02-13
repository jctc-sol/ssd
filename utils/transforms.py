import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as FT
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools import mask
from utils.coordinates import BoundaryCoord, OffsetCoord
from utils.overlap import find_jaccard_overlap


class PhotometricDistort(object):
    """
    (Ref: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py)
    Distort brightness, contrast, saturation, and hue, each with a probability of `p` chance, in random order.
    :param image: image, a PIL Image
    :return: distorted image
    """
    def __init__(self, p):
        self.proba = p
        self.distortions = [FT.adjust_brightness,
                            FT.adjust_contrast,
                            FT.adjust_saturation,
                            FT.adjust_hue]
        
    def __call__(self, sample):
        img = sample['image']
        random.shuffle(self.distortions)
        for d in self.distortions:
            if random.random() < self.proba:
                if d.__name__ == 'adjust_hue':
                    # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                    adjust_factor = random.uniform(-18 / 255., 18 / 255.)
                else:
                    # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                    adjust_factor = random.uniform(0.5, 1.5)
                # Apply this distortion
                sample['image'] = d(sample['image'], adjust_factor)
        return sample
    
    
class Zoomout(object):
    """
    (Ref: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py)
    Perform a zooming out operation by placing the image in a larger canvas of filler material.
    Helps to learn to detect smaller objects.
    """
    def __init__(self, p, max_scale=4):
        self.proba = p
        self.max_scale = max_scale
        
        
    def __call__(self, sample):
        image = sample['image']
        boxes = sample['boxes']        
        if random.random() < self.proba:            
            _, original_h, original_w = image.size()
            scale = random.uniform(1, self.max_scale)
            new_h = int(scale * original_h)
            new_w = int(scale * original_w)

            # Create such an image with the filler
            filler  = torch.FloatTensor([image[0,:,:].mean(), image[1,:,:].mean(), image[2,:,:].mean()])  # (3)
            new_img = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
            # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
            # because all expanded values will share the same memory, so changing one pixel will change all

            # Place the original image at random coordinates in this new image (origin at top-left of image)
            left = random.randint(0, new_w - original_w)
            right = left + original_w
            top = random.randint(0, new_h - original_h)
            bottom = top + original_h
            new_img[:, top:bottom, left:right] = image

            # Adjust bounding boxes' coordinates accordingly
            # (n_objects, 4), n_objects is the no. of objects in this image
            new_boxes = boxes + torch.FloatTensor([left, top, 0, 0]).unsqueeze(0)
            
            sample['image'] = new_img
            sample['boxes'] = new_boxes
        return sample
    
    
class Flip(object):
    """
    (Ref: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py)
    Perform a flip of the image about the vertical axis of the image (i.e. horizontal flip)
    """
    def __init__(self, p):
        self.proba = p
        
        
    def __call__(self, sample):
        if random.random() < self.proba:
            image = sample['image']
            boxes = sample['boxes']
            # flip image
            sample['image'] = FT.hflip(image)
            # flip boxes                    
            boxes[:,0] = image.width - boxes[:,0] - 1 - boxes[:,2]
            sample['boxes'] = boxes
        return sample
    
    
class Resize(object):
    """
    Resize image and bounding boxes to specified target `size` in the 
    form of either integer or tuple (H, W)
    """
    def __init__(self, size):
        self.target_size = size
        

    def __call__(self, sample):
        # original image
        image = sample['image']                
        height, width = image.size()[1], image.size()[2]
        # resize image
        new_image = FT.resize(image, self.target_size)
        new_height, new_width = new_image.size()[1], new_image.size()[2]
        # resize boxes
        boxes = sample['boxes']    
        boxes[:,0] = boxes[:,0] * new_width / width
        boxes[:,1] = boxes[:,1] * new_height / height
        boxes[:,2] = boxes[:,2] * new_width / width
        boxes[:,3] = boxes[:,3] * new_height / height
        sample['image'] = new_image
        sample['boxes'] = boxes
        return sample
        

class Normalize(object):
    """
    Normalizes the image tensor(s) (expected to be within the range [0,1]) 
    of dimensions [C x W x H] with specified `mean` and `std`. The default 
    values of `mean` and `std` are based on torchvision pretrained models
    as specified here: https://pytorch.org/docs/stable/torchvision/models.html
    """    
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std  = std
        
        
    def __call__(self, sample):
        image = sample['image']
        sample['image'] = FT.normalize(image, self.mean, self.std)
        return sample


    def decode(self, image):
        """
        image: single [C, W, H] image
        """
        for img, mean, std in zip(image, self.mean, self.std):
            img.mul_(std).add_(mean)
        return image


class ImageToTensor(object):
    """
    Provides encode/decode methods to transform PILimage to tensor and vice-versa.
    The forward call of the class method assumes the input is of type `dict` with 'image'
    as one of its keys that holds a corresponding value that is a PILimage, and performs
    `encode` on the image.
    """
    def __init__(self):
        self.encoder = transforms.ToTensor()
        self.decoder = transforms.ToPILImage()
        
    def __call__(self, sample):
        sample['image'] = self.encode(sample['image'])
        return sample

    def encode(self, img):
        return self.encoder(img)
    
    def decode(self, tensor):
        return self.decoder(tensor)
    
    
class CategoryToTensor(object):
    """
    Provides encodes/decodes methods to transform category classes from 
    np.array format to torch.tensor format and vice-versa. 
    The forward call of the class method assumes the input is of type `dict`
    with 'cats' as one of its keys that holds a corresponding value that is 
    a np.array, and performs `encode` on the numpy array.
    """        
    def __call__(self, sample):
        sample['cats'] = self.encode(sample['cats'])
        return sample

    def encode(self, cats):
        return torch.LongTensor(cats)
    
    def decode(self, tensor):
        return tensor.numpy()
    
    
class BoxToTensor(object):
    """
    Provides encodes/decodes methods to transform bounding boxes from 
    np.array format to torch.tensor format and vice-versa. 
    The forward call of the class method assumes the input is of type `dict`
    with 'boxes' as one of its keys that holds a corresponding value that is 
    a np.array, and performs `encode` on the numpy array.
    """       
    def __call__(self, sample):
        sample['boxes'] = self.encode(sample['boxes'])
        return sample

    def encode(self, boxes):
        return torch.FloatTensor(boxes)
    
    def decode(self, tensor):
        return tensor.numpy()

    
class CocoBoxToFracBoundaryBox(object):
    """
    Convert bounding box coordinates from COCO dataset format (x,y,w,h)
    to fractional boundary box coordinates (x_min,y_min,x_max,y_max)
    where each value is normalized between [0,1] to the height and width
    of the image.
    """
    def __init__(self):
        pass
    
    def __call__(self, sample):
        boxes = sample['boxes']
        height, width = sample['dims'][0], sample['dims'][1]
        sample['boxes'] = self.encode(height, width, boxes)
        return sample
    
    def encode(self, height, width, boxes):             
        # convert COCO coordinates to fractional bounding box coordinates 
        # with coordinate values [0,1]
        boxes[:,2] = boxes[:,0] + boxes[:,2]
        boxes[:,3] = boxes[:,1] + boxes[:,3]
        # normalize coordinates by image dims
        boxes[:,0] = boxes[:,0] / width
        boxes[:,1] = boxes[:,1] / height
        boxes[:,2] = boxes[:,2] / width
        boxes[:,3] = boxes[:,3] / height        
        return boxes
    
    def decode(self, height, width, boxes):
        # convert fractional bounding box coordinates back to COCO coordinates
        # convert scale 
        boxes[:,0] = boxes[:,0] * width
        boxes[:,1] = boxes[:,1] * height
        boxes[:,2] = boxes[:,2] * width
        boxes[:,3] = boxes[:,3] * height
        # get height and width
        boxes[:,2] = boxes[:,2] - boxes[:,0]
        boxes[:,3] = boxes[:,3] - boxes[:,1]
    
    
class AssignObjToPbox(object):
    """
    Assigns each ground truth object in the image to the best prior bounding box
    of the network, based on best intersection over union (IoU, aka Jaccard 
    overlap)
    """
    def __init__(self, pboxes, threshold):
        self.pboxes_bc = pboxes
        self.n_pboxes  = pboxes.size()[1]
        self.boundaryCoord = BoundaryCoord()
        self.offsetCoord   = OffsetCoord()
        self.threshold = threshold
        
    def __call__(self, sample):
        # get ground truth box coordinates (in boundary coordinates), and
        # ground truth object class labels        
        true_boxes_bc = sample['boxes']
        true_classes  = sample['cats']
        true_offset, true_cls = self.assign(true_boxes_bc, true_classes)        
        # add the true pbox offset and true pbox classes to sample dictionary
        sample['pbox_offset'] = true_offset
        sample['pbox_cls'] = true_cls
        return sample
        
    def assign(self, true_boxes_bc, true_classes):
        # number of ground truth objects to assign
        n_objs = true_boxes_bc.size(0)
        
        # init zero contains for true location offset and classes
        true_offset = torch.zeros((self.n_pboxes, 4), dtype=torch.float)
        true_cls    = torch.zeros((self.n_pboxes),    dtype=torch.long)

        # ---------------------------------------------------------------------------------
        # i. compute IoU overlaps between object boxes and prior boxes
        # ---------------------------------------------------------------------------------
        # find overlap of each ground truth objects with each of the prior bboxes
        # (note: that both set of boxes need to be in boundary-coordinate format)
        iou = find_jaccard_overlap(true_boxes_bc, self.pboxes_bc)  # (n_objects, 8732)

        # ---------------------------------------------------------------------------------
        # ii. find best pbox for each object based on best IoU & suppress background objects
        # ---------------------------------------------------------------------------------
        # for each prior boxes, find the ground truth object that overlaps the most with it
        obj_overlap_with_pbox, obj_assigned_to_pbox = iou.max(dim=0)  # (8732)
        # conversely, for each object, find the prior boxes that ooverlaps the most with it
        _, best_pbox_for_obj = iou.max(dim=1)  # (n_objects)

        # ---------------------------------------------------------------------------------
        # iii. fix potential issues due to poor overlaping between object & prior boxes
        # ---------------------------------------------------------------------------------
        # one potential problem is poor overlap between object and any prior boxes
        # in this case, make sure each object is assign to the prior boxes with most overlap
        obj_assigned_to_pbox[best_pbox_for_obj] = torch.LongTensor(range(n_objs))
        # get object class label for each prior bounding box
        obj_cls_for_pbox = true_classes[obj_assigned_to_pbox]

        # also ensure each object is captured by at least one prior box; manually change 
        # the overlap to 1 to avoid it getting assigned to background object after thresholding
        obj_overlap_with_pbox[best_pbox_for_obj] = 1.
        # for all other pboxes with overlap less than threshold, assign them to background object
        obj_cls_for_pbox[obj_overlap_with_pbox < self.threshold] = 0

        # ---------------------------------------------------------------------------------
        # iv. record the ground truths for this image
        # ---------------------------------------------------------------------------------        
        # compute the offset coordinates of object locations relative to prior boxes
        true_boxes_cc  = self.boundaryCoord.decode(true_boxes_bc)
        pboxes_cc      = self.boundaryCoord.decode(self.pboxes_bc)
        true_offset    = self.offsetCoord.encode(true_boxes_cc[obj_assigned_to_pbox], pboxes_cc)
        # add true object class label allocation for each prior bounding box
        true_cls = obj_cls_for_pbox       
        
        return true_offset, true_cls