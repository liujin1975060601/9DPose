                                          
"""
Model validation metrics
"""

import math
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import DOTA_devkit.polyiou.polyiou as polyiou


def fitness(x):
                                                        
    w = [0.0, 0.0, 0.1, 0.9]                                             
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
              
             
                 
                      
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

                        
    i = np.argsort(-conf)              
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]           
                                      

                         
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]                                           

                                                                 
    px, py = np.linspace(0, 1, 1000), []                
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    f1 = np.zeros((nc, 1000))
    ic = np.zeros(nc)
    theshes = torch.ones(len(names))
                                      
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()                    
        n_p = i.sum()                         

        if n_p == 0 or n_l == 0:
                                  
            continue
        else:
                                    
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

                    
            recall = tpc / (n_l + 1e-16)                
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)                                       

                       
            precision = tpc / (tpc + fpc)                   
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)                 

                                            
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))                        
                                                                
            f1[ci] = 2 * p[ci] * r[ci] / (p[ci] + r[ci] + 1e-16)            
            ic[ci] = f1[ci].argmax()        
            theshes[int(c)] = ic[ci] / px.shape[0]

                                                        
                                                 
    names = [v for k, v in names.items() if k in unique_classes]                                     
    names = {i: v for i, v in enumerate(names)}           
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

                
    ic = np.round(ic).astype(int)
                           
    return p[np.arange(nc), ic], r[np.arange(nc), ic], ap, f1[np.arange(nc), ic], unique_classes.astype('int32'), theshes
    '''
    idf1 = f1.mean(0).argmax()  # max F1 index
    return p[:, idf1], r[:, idf1], ap, f1[:, idf1], unique_classes.astype('int32'), theshes
    '''


def ap_per_class_dir(tp, conf, pred_cls, target_cls, dir_tp, plot=False, save_dir='.', names=()):

                        
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    dir_tp = dir_tp[i]

                         
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]                                           

                                                                 
    px, py = np.linspace(0, 1, 1000), []                
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))

    dtps = np.zeros(len(unique_classes), dtype=np.int)
    dfps = np.zeros(len(unique_classes), dtype=np.int)
    dtpfns = np.zeros(len(unique_classes), dtype=np.int)

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()                    
        n_p = i.sum()                         

        if n_p == 0 or n_l == 0:
            continue
        else:
                                    
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)
                 
            dfps[ci] = (1 - dir_tp[i]).sum()
            dtps[ci] = dir_tp[i].sum()
            dtpfns[ci] = n_l
                    
            recall = tpc / (n_l + 1e-16)                
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)                                       

                       
            precision = tpc / (tpc + fpc)                   
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)                 

                                            
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))                        

                                                        
    f1 = 2 * p * r / (p + r + 1e-16)
    names = [v for k, v in names.items() if k in unique_classes]                                     
    names = {i: v for i, v in enumerate(names)}           
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    cls_dir_acc = dtps / (dfps + dtpfns)
    i = f1.mean(0).argmax()                
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32'), cls_dir_acc



def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

                                                 
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

                                    
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

                                
    method = 'interp'                                   
    if method == 'interp':
        x = np.linspace(0, 1, 101)                           
        ap = np.trapz(np.interp(x, mrec, mpre), x)             
    else:                
        i = np.where(mrec[1:] != mrec[:-1])[0]                                        
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])                    

    return ap, mpre, mrec


class ConfusionMatrix:
                                                                                      
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc                     
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1           
            else:
                self.matrix[self.nc, gc] += 1                 

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1                 

    def process_batch_poly(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2, x3, y3, x4,y4) format.
        Arguments:
            detections (Array[N, 10]), x1, y1, x2, y2, x3, y3, x4,y4 conf, class
            labels (Array[M, 9]), class, x1, y1, x2, y2, x3, y3, x4,y4
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 8] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 9].int()
        iou = poly_iou(labels[:, 1:], detections[:, :8])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1           
            else:
                self.matrix[self.nc, gc] += 1                 

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1                 

    def matrix(self):
        return self.matrix

    def plot(self, normalize=True, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)                     
            array[array < 0.005] = np.nan                                         

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)                  
            labels = (0 < len(names) < 99) and len(names) == self.nc                             
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')                                                                   
                sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
            plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
                                                             
    box2 = box2.T

                                           
    if x1y1x2y2:                         
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:                               
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

                       
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) *\
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

                
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)                                         
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)                 
        if CIoU or DIoU:                                                               
            c2 = cw ** 2 + ch ** 2 + eps                           
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4                           
            if DIoU:
                return iou - rho2 / c2        
            elif CIoU:                                                                                      
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)        
        else:                                             
            c_area = cw * ch + eps               
            return iou - (c_area - union) / c_area        
    else:
        return iou       



def ab_iou(ab1, ab2,eps=1e-7):
    
    inter = torch.min(ab1, ab2).prod(1)
    iou =  inter / ( ab1.prod(1) + ab2.prod(1) - inter + eps)
    return iou


def box_iou(box1, box2):
                                                                            
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
                   
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

                                                           
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)                                         


def poly_iou(poly1, poly2):
    device = poly1.device
    poly1 = poly1.cpu().numpy().astype(np.float64)
    poly2 = poly2.cpu().numpy().astype(np.float64)
    polys_1 = []
    polys_2 = []
    for i in range(len(poly1)):
        tm_polygon = polyiou.VectorDouble([poly1[i][0], poly1[i][1],
                                           poly1[i][2], poly1[i][3],
                                           poly1[i][4], poly1[i][5],
                                           poly1[i][6], poly1[i][7]])
        polys_1.append(tm_polygon)

    for i in range(len(poly2)):
        tm_polygon = polyiou.VectorDouble([poly2[i][0], poly2[i][1],
                                        poly2[i][2], poly2[i][3],
                                        poly2[i][4], poly2[i][5],
                                        poly2[i][6], poly2[i][7]])
        polys_2.append(tm_polygon)
    
    
    n = len(poly1)
    m = len(poly2)
    ious = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            iou = polyiou.iou_poly(poly1[i], poly2[j])
            ious[i][j] = iou
    return torch.from_numpy(ious).to(device)

    
    


def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

                                           
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

                       
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) *\
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

               
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

                                 
    return inter_area / box2_area


def abc_iou(anchorsabc, tabc):
                                                              
    anchorsabc = anchorsabc[:, None]            
    tabc = tabc[None]            
    inter_sec = torch.min(anchorsabc, tabc)             
    inter = inter_sec.prod(2)          
    return inter / (anchorsabc.prod(2) + tabc.prod(2) - inter)                                         


                                                                                                                        

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
                            
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:                                            
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')                           
    else:
        ax.plot(px, py, linewidth=1, color='grey')                           

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
                             
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:                                            
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')                            
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')                            

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()
