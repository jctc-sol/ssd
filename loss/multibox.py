import torch
from utils.coordinates import BoundaryCoord, OffsetCoord
from utils.overlap import find_jaccard_overlap
from torch import nn


class Loss(nn.Module):
    """
    Loss funcion for object detection, which is a linear combination of:
    a) object localization loss for the predicted bounding box location; and
    b) classification loss for the predicted object class
    
    Code reference: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py#L532
    """
    
    def __init__(self, neg_pos_ratio=3, alpha=1., loc_loss_tracker=None, cls_loss_tracker=None, device=None):
        """
        :param pboxes: prior bounding boxes of object detection model, provided in center coordinates
        :param threshold: cutoff threshold on IoU overlap between a pair of true object box and prior bounding box
        :param neg_pos_ratio: ratio to be used in hard negative sample minning
        :param alpha: relative weighting between localization & classification losses
        :param loc_loss_tracker: scalar value tracker to track the loss history for location loss
        :param cls_loss_tracker: scalar value tracker to track the loss history for classification loss 
        """
        super(Loss, self).__init__()        

        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.device = device
        # localization/classification losses
        self.loc_loss = nn.L1Loss()
        self.cls_loss = nn.CrossEntropyLoss(reduction='none')
        self.loc_loss_tracker = loc_loss_tracker
        self.cls_loss_tracker = cls_loss_tracker
        
        
    def forward(self, pred_boxes, pred_scores, true_locs, true_cls):
        """
        Forward pass to compute the loss given predicted bounding boxes and predicted classification scores
        from an object detection model. N for batch size below.
        :param pred_boxes:  predicted bound boxes from object detection model in offset coordinates form; tensor of dim (N, 8732, 4)
        :param pred_scores: predicted classification scores from boject detection model; tensor of dim (N, 8732, n_classes)
        :param true_locs: true object offsets wrt prior bounding boxes
        :param true_cls: true object class label assigned to each prior bounding boxes
        
        :return: scalar loss measure
        """        
        bs = pred_boxes.size(0)
        n_priors  = true_locs[0].size(1)
        n_classes = pred_scores.size(-1)
        assert n_priors == pred_boxes.size(1) == pred_scores.size(1)

        true_locs = torch.cat(true_locs, dim=0)
        true_cls  = torch.cat(true_cls, dim=0)
                
        # ---------------------------------------------------------------------------------
        # 1. LOCALIZATION LOSS of non-background objects
        # ---------------------------------------------------------------------------------
        # get flag for all non-background prior bounding boxes (i.e. class label > 0)
        nonbackground_priors = true_cls != 0  # bit map of size (N, 8732)
        loc_loss = self.loc_loss(pred_boxes[nonbackground_priors], true_locs[nonbackground_priors])
        
        # ---------------------------------------------------------------------------------
        # 2. CLASSIFICATION LOSS of non-background objects
        # ---------------------------------------------------------------------------------
        cls_loss_all = self.cls_loss(pred_scores.view(-1, n_classes), true_cls.view(-1))
        cls_loss_all = cls_loss_all.view(bs, n_priors)  # reshape back to size (N, 8732)
        cls_loss_pos = cls_loss_all[nonbackground_priors].sum()

        # ---------------------------------------------------------------------------------
        # 3. Hard Negative Mining
        # ---------------------------------------------------------------------------------
        # HNM is used in the case where there is a large imbalance between -v vs +ve class.
        # This is often the case in object detection since the majority of bounding boxes 
        # would capture background objects (class = 0). Artificially balance out the 
        # -ve vs +ve class ratio by selecting `n` number of -ve samples with the largest loss 
        # (i.e. hardest -v samples)
        if self.neg_pos_ratio > 0:        
            # get number of hard negatives to sample
            n_positives   = nonbackground_priors.sum(dim=1).sum().float()
            n_neg_samples = self.neg_pos_ratio * n_positives

            # set positive prior losses to 0 since we've already computed cls_loss_pos
            cls_loss_neg = cls_loss_all.clone()
            cls_loss_neg[nonbackground_priors] = 0.
            # sort losses in decending order
            cls_loss_neg, _ = cls_loss_neg.sort(dim=1, descending=True)
            hardness_ranks  = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(cls_loss_neg).to(self.device)  # (N, 8732)
            # get the hard -ve samples and sum of their losses
            hard_neg = hardness_ranks < n_neg_samples.unsqueeze(-1) # (N, 8732)
            cls_loss_hard_neg = cls_loss_neg[hard_neg].sum()
        
            # COMBINED CLASSIFICATION LOSS
            # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
            # and averaged across number of possible classes
            cls_loss = (cls_loss_pos + cls_loss_hard_neg) / n_positives / n_classes
        else:
            cls_loss = cls_loss_pos

        if self.loc_loss_tracker: self.loc_loss_tracker.update(loc_loss.item())
        if self.cls_loss_tracker: self.cls_loss_tracker.update(cls_loss.item())
        print(f'location loss: {loc_loss.item()}')
        print(f'class loss: {cls_loss.item()}')
        # total loss
        return loc_loss + self.alpha * cls_loss

