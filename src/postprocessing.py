import torch
from torch import nn
import torch.nn.functional as F
#from imutils import perspective закоментил
import numpy as np
import cv2

def _nms( heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep

def _gather_feat(feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

def topk(scores, K = None):
    scores = _nms(scores)
    
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), 11)
    
    if(K==None):
        K = torch.sum(topk_scores > 0.2)
    
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    
    topk_inds = _gather_feat( topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    
    inds, sort_ids = torch.sort(topk_inds, dim=1)
    
    return sort_ids.detach().cpu().numpy() ,topk_ys.detach().cpu().numpy().astype('int'), topk_xs.detach().cpu().numpy().astype('int')


def get_coordinates_hm(out):
    inds_c, yc, xc = topk(out[:,0:1,:,:])
    inds_tl, tl_y, tl_x = topk(out[:,1:2,:,:], len(yc[0]))
    inds_tr, tr_y, tr_x = topk(out[:,2:3,:,:], len(yc[0]))
    inds_bl, bl_y, bl_x = topk(out[:,3:4,:,:], len(yc[0]))
    inds_br, br_y, br_x = topk(out[:,4:5,:,:], len(yc[0]))
    
    bboxes = []
    
    for i in range(len(tl_y)):
        xc[i] = xc[i][inds_c[i]]
        yc[i] = yc[i][inds_c[i]]
        
        tl_x[i] = tl_x[i][inds_tl[i]]
        tl_y[i] = tl_y[i][inds_tl[i]]
        
        tr_x[i] = tr_x[i][inds_tr[i]]
        tr_y[i] = tr_y[i][inds_tr[i]]
        
        bl_x[i] = bl_x[i][inds_bl[i]]
        bl_y[i] = bl_y[i][inds_bl[i]]
        
        br_x[i] = br_x[i][inds_br[i]]
        br_y[i] = br_y[i][inds_br[i]]

    
    min_len = np.min([len(xc[0]), len(tl_x[0]), len(tr_x[0]), len(bl_x[0]), len(br_x[0])])
    
    for i in range(min_len):
        bboxes.append(np.array([[tl_x[0][i], tl_y[0][i]], [tr_x[0][i], tr_y[0][i]],[br_x[0][i], br_y[0][i]],[bl_x[0][i], bl_y[0][i]]]))
    
    cts = []
    for i in range(len(xc[0])):
        cts.append(np.array([xc[0][i], yc[0][i]]))
    return np.array(bboxes), np.array(cts)

def annotation_format(bboxes):
    outputs = []
    bbs = []
    for box in bboxes:
        (xc_, yc_), (w, h), angle = cv2.minAreaRect(box)
        bbs.append(cv2.boxPoints(((xc_,yc_), (w, h), angle)))
        outputs.append(((xc_, yc_), (w,h), angle))
    return outputs, np.array(bbs)


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """
    
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)
