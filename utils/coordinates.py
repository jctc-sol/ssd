import torch


class BoundaryCoord():
    """
    Encodes/decodes the bounding box Center coordinates (x_c, y_c, w_c, h_c) to/from Boundary coordinates 
    (x_1, y_1, x_2, y_2) where (x_1, y_1) specifies the upper-left corner and (x_2, y_2) the lower-right
    corner of the boundary of bounding boxes
    """   
    def __init__(self):
        pass
    
    def encode(self, boxes):
        """
        Encodes bounding boxes tensor in Center coordinates to Boundary coordinates.
        
        boxes: bounding boxes tensor in center coordinates (x_c, y_c, w_c, h_c) format
        return: bounding boxes tensor in boundary coordinates (x_1, y_1, x_2, y_2) format
        """
        w_c, h_c = boxes[:,2], boxes[:,3]
        x1 = boxes[:,0] - w_c/2.0
        y1 = boxes[:,1] - h_c/2.0
        x2 = boxes[:,0] + w_c/2.0
        y2 = boxes[:,1] + h_c/2.0
        coords = [x1, y1, x2, y2]
        return torch.clip(torch.cat([c.unsqueeze(-1) for c in coords], dim=-1), 0, 1)
        
    def decode(self, boxes):
        """
        Decodes bounding boxes tensor in Boundary coordinates to Center coordinates.
        
        boxes: bounding boxes tensor in boundary coordinates (x_1, y_1, x_2, y_2) format
        return: bounding boxes tensor in center coordinates (x_c, y_c, w_c, h_c) format
        """
        w_c = boxes[:,2] - boxes[:,0]
        h_c = boxes[:,3] - boxes[:,1]
        x_c = boxes[:,0] + w_c/2.0
        y_c = boxes[:,1] + h_c/2.0
        coords = [x_c, y_c, w_c, h_c]
        return torch.cat([c.unsqueeze(-1) for c in coords], dim=-1)
    

class OffsetCoord():
    """
    Encodes/decodes the center coordinates (x_c, y_c, w_c, h_c) of bounding boxes relative to the prior 
    boxes (from SSD, expressed also in center coordinates) in terms of offset coordinates. This offset 
    coordinates is the form that is output by the SSD locator prediction. The offset coordinates have 
    the following relation:
    (dx, dy) = ((x_c - x_p)/(x_p/10), (y_c - y_p)/(y_p/10)); and 
    (dw, dh) = (log(w_c/(w_p*5)), log(h_c/(h_p*5)))
    """
    def __init__(self):
        pass

        
    def encode(self, cxcy, priors_cxcy):        
        """
        converts cxcy (center coordinates) to oxoy (offset coordinates)        
        cxcy: bounding box in center-coordinate format
        prior_cxcy: prior box in center-coordinate format
        """
        oxoy = (cxcy[:,:2] - priors_cxcy[:,:2]) / priors_cxcy[:,2:]
        owoh = torch.log(cxcy[:,2:] / priors_cxcy[:,2:])
        return torch.cat([oxoy, owoh], dim=1)
    
    
    def decode(self, oxoy, priors_cxcy):
        """
        converts oxoy (offset coordinates) back to cxcy (center coordinates)
        oxoy: bounding boxes in offset-coordinate format wrt SSD's prior bounding boxes
        """
        cxcy = oxoy[:,:2] * priors_cxcy[:,2:] + priors_cxcy[:,:2]
        cwch = torch.exp(oxoy[:,2:]) * priors_cxcy[:,2:]
        return torch.cat([cxcy, cwch], dim=1)