import torch
from torch import nn
from model.base import VGGBase
from model.auxiliary import AuxLayers
from model.prediction import PredLayers
from model.utils import add_module, create_pboxes
from utils.coordinates import BoundaryCoord, OffsetCoord
from utils.overlap import find_jaccard_overlap


class AdaptiveScaler(nn.Module):
    
    def __init__(self, n, init_scale):
        super(AdaptiveScaler, self).__init__()
        
        # lower level features (ftmap4) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but it is learnable for each channel during back-prop
        self.rescale = nn.Parameter(torch.FloatTensor(1, n, 1, 1))
        nn.init.constant_(self.rescale, init_scale)
        

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()
        x = x / norm * self.rescale
        return x
        

class SSD300(nn.Module):
    
    def __init__(self, n_classes, device=None):
        super(SSD300, self).__init__()
        
        self.device = device
        self.n_classes = n_classes
        
        # network components
        self.base = VGGBase()
        self.rescaler = AdaptiveScaler(512, 20)
        self.aux  = AuxLayers()
        self.pred = PredLayers(self.n_classes)
        
        # create prior bounding boxes
        self.pboxes = create_pboxes()
        self.pboxes.to(self.device)
        
        # instantiate a coordinate transformation object to decipher object location
        # output in prior box offset coordinate format to center coordinate format
        self.oc2cc = OffsetCoord()
        # instantiate a coordinate transformation object to decipher object location
        # output in center coordinate format to boundary box coordinate format
        self.cc2bc = BoundaryCoord()
        

    def fine_tune(self, freeze=True):
        for param in self.components.base.parameters():
            param.requires_grad = not freeze

            
    def forward(self, image):
        """
        Forward propagation.
        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        # ftmap4: (N, 512, 38, 38), ftmap7: (N, 1024, 19, 19)
        ftmap4, ftmap7 = self.base(image)                     
        
        # Normalize conv4_3 with L2 norm & rescale using the learnable scale factor for each channel
        ftmap4 = self.rescaler(ftmap4)
    
        # Run auxiliary convolutions (higher level feature map generators)
        # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)
        ftmap8, ftmap9, ftmap10, ftmap11 = self.aux(ftmap7)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)        
        # (N, 8732, 4), (N, 8732, n_classes)
        locations, cls_scores = self.pred(ftmap4, ftmap7, ftmap8, ftmap9, ftmap10, ftmap11)

        return locations, cls_scores


    def detect_objects(self, offsets_pred, cls_scores_pred, nms_threshold, cls_score_threshold, top_k):
        """
        Post-process the prediction from the SSD output that applies Non-Maximum Suppression (NMS)
        based on `min_score`, `max_overlap`, and `top_k` criteria to reduce the number of resulting
        bounding boxes.
        
        For each below, let M be `batch_size`, `n_i` is the number of predicted objects in each image, and 
        `N_i` is the number of true objects in each image.
        
        :param offsets_pred: predicted bounding box offset coordinates w.r.t the 8732 prior anchor 
                             boxes, a tensor of dimensions (M, 8732, 4)
        :param cls_scores_pred: predicted class scores for each of the 8732 prior anchor/bounding box locations, 
                                a tensor of dimensions (M, 8732, n_classes)
        :param nms_threshold: non-max suppression threshold for eliminating overlapping canidate bounding regions
        :param cls_score_threshold: minimum score threshold to apply against the class score to be considered a 
                                    match for a certain class
        :param top_k: if there are a lot of resulting detection boxes, keep only the top 'k'

        :return: detected_boxes: M length list of tensors (n_i, 4) for detected bounding boxes after NMS
        :return: detected_labels: M length list of labels (n_i, n_classes) for detected class labels
        :return: detected_scores: `batch_size` length list of scores (n_i, n_classes) 
        
        Source ref: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py#L426
        """
        batch_size = offsets_pred.size(0)
        n_priors   = self.pboxes.size(0)
        # apply softmax to the prediction class scores
        cls_scores_pred = nn.functional.softmax(cls_scores_pred, dim=2)

        # ensure # of prior boxes align across input location & score predictions
        assert n_priors == offsets_pred.size(1) == cls_scores_pred.size(1),\
        "number of prior boxes, offset location prediction & bbox class prediction mismatch"

        # Lists to store final predicted boxes, labels, and scores for all images
        detected_boxes  = list()
        detected_labels = list()
        detected_scores = list()

        # iterate through each image in the batch
        for i in range(batch_size):
            
            # Init empty lists to store boxes and scores for this image
            image_boxes  = list()
            image_labels = list()
            image_scores = list()
            
            # model output of predicted boxes are natively in offset coordinate format,
            # first decode it back to center box coordinate format, then encode from 
            # center box coordinate to boundary coordinate format
            predicted_boxes_bc = self.cc2bc.encode(
                # decode predicted_boxes from offset coordinates to center coordinates
                # note that self.prior_boxes are in center coordinates
                self.oc2cc.decode(offsets_pred[i], self.pboxes)
            )  # size (8732, 4)
            
            # determine the most probable class & score from the softmax of cls_scores_pred
            max_scores, pedicted_labels = cls_scores_pred[i].max(dim=1)  # size (8732)


            # iterate through each class (except for class 0 which is reserved for background)
            for c in range(1, self.n_classes):
                
                # get all scores belonging to this class that are above score threshold at each 
                # prior bounding box locations
                class_scores = cls_scores_pred[i][:, c]  # size (8732)
                # note: torch.uint8 (byte) tensor; this can be used to locate the set of 
                # prior bounding boxes with class score above threshold
                score_above_min_score = class_scores > cls_score_threshold
                # skip remainder steps if there are no scores above threshold
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                # get scores & associated locator predictions
                class_scores = class_scores[score_above_min_score]        # size (n_above_min_score)
                class_boxes  = predicted_boxes_bc[score_above_min_score]  # size (n_above_min_score, 4)
                # sort according to score from highest to lowest
                class_scores, sort_idx = class_scores.sort(dim=0, descending=True)
                class_boxes = class_boxes[sort_idx]
                
                # compute jaccard overlap between all class boxes; returns
                # size (n_above_min_score, n_above_min_score)
                overlap = find_jaccard_overlap(class_boxes, class_boxes)

                # Non-Maximum Suppression (NMS)
                # init a torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppression_idx = torch.zeros((n_above_min_score), dtype=torch.uint8).to(self.device)  # (n_qualified)

                # Iterate through each box in order of most confident scores
                for box in range(class_boxes.size(0)):
                    # If this box is already marked for suppression
                    if suppression_idx[box] == 1:
                        continue
                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppression_idx = torch.max(suppression_idx, overlap[box] > nms_threshold)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation
                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppression_idx[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_boxes[1 - suppression_idx])
                image_labels.append(torch.LongTensor((1 - suppression_idx).sum().item() * [c]).to(self.device))
                image_scores.append(class_scores[1 - suppression_idx])



            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(self.device))
                image_labels.append(torch.LongTensor([0]).to(self.device))
                image_scores.append(torch.FloatTensor([0.]).to(self.device))

            # Concatenate into single tensors
            image_boxes  = torch.cat(image_boxes, dim=0)  # (n_qualified, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_qualified)
            image_scores = torch.cat(image_scores, dim=0)  # (n_qualified)
            n_objects = image_scores.size(0)

            # Keep only the top k highest score objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes  = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            detected_boxes.append(image_boxes)
            detected_labels.append(image_labels)
            detected_scores.append(image_scores)

        return detected_boxes, detected_labels, detected_scores  # lists of length batch_size
