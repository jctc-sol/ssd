from exp.callbacks.core import Callback


class CocoDataBatchCallback(Callback):
    
    def __init__(self):
        super(CocoDataBatchCallback, self).__init__()
        
        
    def before_batch(self):        
        self.exp.xb = self.exp.batch['images'].to(self.exp.device)
        self.exp.yb = (self.exp.batch['pbox_offsets'], self.exp.batch['pbox_classes'])
        
        
    def after_pred(self):
        pred_box_offsets  = self.exp.pred[0]
        pred_cls_scores = self.exp.pred[1]
        true_box_offsets  = [b.to(self.exp.device) for b in self.exp.yb[0]]
        true_cls_cats   = [c.to(self.exp.device) for c in self.exp.yb[1]]
        # notify the experiment module to use custom loss since the returned
        # prediction and class label format is different from conventional form
        self.exp.custom_loss = True
        self.exp.loss = self.loss_func(pred_box_offsets, pred_cls_scores, true_box_offsets, true_cls_cats)
