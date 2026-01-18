import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
import cv2
import os
import numpy as np
import random
import torch
from utils.general import pts2dir,dirab2WH
import DOTA_devkit.polyiou.polyiou as polyiou
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_image_label(patches,aug_labels,ax):
    for label in aug_labels:
              
        rect = patches.Rectangle((label[1], label[2]), label[3]-label[1], label[4]-label[2], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)               
              
        polygon = patches.Polygon([(label[5], label[6]), (label[7], label[8]), (label[9], label[10]), (label[11], label[12])], linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(polygon)

def out_range_count(points, W, H):
    sum=0
    for point in points:
        if(point.xx.shape[0]!=4):
            print(point)
        sum += np.count_nonzero(point.xx<0) + np.count_nonzero(point.xx>W) + np.count_nonzero(point.yy<0) + np.count_nonzero(point.yy>H) > 0
    return sum

def cross_adjust(polys):
    for point in polys:
        if(point.xx.shape[0]==4):
            x0,x1,x2,x3 = point.xx[0],point.xx[1],point.xx[2],point.xx[3]
            y0,y1,y2,y3 = point.yy[0],point.yy[1],point.yy[2],point.yy[3]
            CrossProduct_01_12 = (x1 - x0) * (y2 - y1) - (x2 - x1) * (y1 - y0)
            CrossProduct_12_23 = (x2 - x1) * (y3 - y2) - (x3 - x2) * (y2 - y1)
            if(bool(CrossProduct_01_12>0) ^ bool(CrossProduct_12_23>0)):           
                point.xx[2],point.xx[3],point.xx_int[2],point.xx_int[3] = x3, x2, np.round(x3), np.round(x2)
                point.yy[2],point.yy[3],point.yy_int[2],point.yy_int[3] = y3, y2, np.round(y3), np.round(y2)
                             
        else:
            print(point)
def cal_poly_area(pts8):
    points = pts8.reshape(4,2)
                 
    points = np.vstack((points, points[0]))

                      
    clockwise_sum = np.sum(points[:-1, 0] * points[1:, 1])
    counterclockwise_sum = np.sum(points[:-1, 1] * points[1:, 0])

             
    area = abs(clockwise_sum - counterclockwise_sum) / 2
    return area
def out_range_filt(polys, shape, iou_thresh=0.5):
    image_poly = np.array([0, 0, shape[0], 0, shape[0],shape[1], 0,shape[1]], dtype=np.float64)              
    image_area = shape[0]*shape[1]
                                            
    polys_out=[]
    for point in polys:
        if(point.xx.shape[0]==4):
            poly_pts = np.array([point.xx[0], point.yy[0], point.xx[1],point.yy[1], point.xx[2],point.yy[2], point.xx[3],point.yy[3]], dtype=np.float64)
            iou = polyiou.iou_poly(poly_pts, image_poly)
            obj_area = cal_poly_area(poly_pts)
            sec_area = iou * (image_area + obj_area) / (1 + iou)
            if 1:
                if(sec_area / obj_area > iou_thresh):
                    polys_out.append(point)
                else:
                    pass             
            else:
                cx,cy = point.xx.mean(),point.yy.mean()
                if(cx>=0 and cx<=shape[1] and cy>=0 and cy<=shape[0]):
                    polys_out.append(point)
            
        else:
            print(point)
    return polys_out

class ImageAugPoly:
    def __init__(self,adjust_filt=False) -> None:
                                                                                                                  
        self.augments = [seq, oneof_aug, rotateone, shear, crop, blur]           
        self.debug_samples = 20
        self.iou_thresh = 0.5
        self.adjust_filt = adjust_filt

    def augment(self, image, labels, full_name):                     
        polys = []
        shape = labels.shape[-1]                                  
        for label in labels:
            polys.append(Polygon(label[5:13].reshape(4, 2), label=label[0]))
        boxes = PolygonsOnImage(polys, shape=image.shape)
                                                   
                                  
                             
                         
        na = len(self.augments)       
        idx = random.randint(0, na - 1)          
        aug_img, boxes_aug = self.augments[idx](image, boxes)                              
        assert(len(boxes.polygons)==len(boxes_aug.polygons))
                                              
        if self.adjust_filt:
                                             
            boxes_aug.polygons = out_range_filt(boxes_aug.polygons, image.shape, self.iou_thresh)
        else:
                                                                             
            boxes_aug = boxes_aug.remove_out_of_image()                        
                                                                     
        n = len(boxes_aug)
        if n:
            boxes_aug_array = boxes_aug.to_xy_array().reshape(n,2*4)                                         
            dir_targets = pts2dir(torch.from_numpy(boxes_aug_array), fold_angle=1)                     
                                        
            points = []
            bbox = []
            clss = []
            for i in range(len(boxes_aug.polygons)):
                after = boxes_aug.polygons[i]
                npts = len(after.coords)
                if npts == 4:
                    clss.append(after.label)     
                    cx,cy = after.xx.mean(),after.yy.mean()
                    xmin, ymin = np.min(after.xx), np.min(after.yy)                                   
                    xmax, ymax = np.max(after.xx), np.max(after.yy)                                   
                    bbox.append([xmin, ymin, xmax, ymax])
                    points.append(after.coords.reshape(-1))      
            points = np.array(points)
            clss = np.array(clss)
            bbox = np.array(bbox)
            if len(clss):
                clss = clss.reshape(-1, 1)
                aug_labels = np.concatenate((clss, bbox, points), axis=1)
            else:
                aug_labels = np.zeros((0, shape), dtype=np.float32)
        else:
            aug_labels = np.zeros((0, shape), dtype=np.float32)
        
                   
        if self.debug_samples > 0:
            ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
            ax[0].imshow(image[:, :, ::-1])        
            draw_image_label(patches,labels,ax[0])
            ax[1].imshow(aug_img[:, :, ::-1])          
            draw_image_label(patches,aug_labels,ax[1])
            
                       
            aug_path = os.path.dirname(full_name) + '/../augs/'
            if os.path.exists(aug_path):
                file_name, file_extension = os.path.splitext(os.path.basename(full_name))
                plt.savefig(aug_path + '/' + file_name + '.jpg')
                self.debug_samples-=1
            plt.close()

        return aug_img, aug_labels


def augment_poly(image, labels, r=[-45, 45]):
    polys = []
    for label in labels:
        polys.append(Polygon(label[5:].reshape(4, 2), label=label[0]))
    boxes = PolygonsOnImage(polys, shape=image.shape)
    aug_img, boxes_aug = rotate(image, boxes)
                                                                     
    boxes_aug = boxes_aug.remove_out_of_image()

    n = len(boxes_aug)
    if n:
        points = []
        bbox = []
        clss = []
        for i in range(len(boxes_aug.polygons)):
            after = boxes_aug.polygons[i]
            n = len(after.coords)
            if n < 4 or n > 4:
                continue
            xmin, ymin = np.min(after.xx), np.min(after.yy)
            xmax, ymax = np.max(after.xx), np.max(after.yy)
            points.append(after.coords.reshape(-1))
            clss.append(after.label)
            bbox.append([xmin, ymin, xmax, ymax])
        points = np.array(points)
        clss = np.array(clss)
        bbox = np.array(bbox)
        if len(clss):
            clss = clss.reshape(-1, 1)
            aug_labels = np.concatenate((clss, bbox, points), axis=1)
        else:
            aug_labels = np.zeros((0, 13), dtype=np.float32)
    else:
        aug_labels = np.zeros((0, 13), dtype=np.float32)
    return aug_img, aug_labels
    



def read_anno(path, file_name):
    bboxes = []
    with open(os.path.join(path, file_name), 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            segmentation = [int(float(x)) for x in line[:8]]
            pos = []
            for i in range(0, len(segmentation), 2):
                pos.append((segmentation[i], segmentation[i + 1]))
            catgeory = line[8]
            bboxes.append(Polygon(pos, label=catgeory))
    return bboxes


def seq(image, bbs):
    seqe = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)),                                         
        iaa.Affine(
            translate_px={"x": 40, "y": 60},
            scale=(0.5, 1.5),
            mode="edge"
        )                                                                      
    ])
                             
    image_aug, bbs_aug = seqe(image=image, polygons=bbs)
    return image_aug, bbs_aug


def fliplr(image, bbs, rate=1):
    aug = iaa.Fliplr(rate)
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def flipud(image, bbs, rate=1):
    aug = iaa.Flipud(rate)
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def flipone(image, bbs, rate=0.5):
    aug = iaa.OneOf([
        iaa.Fliplr(rate),
        iaa.Flipud(rate)
    ])
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def rotateone(image, bbs):
    aug = iaa.OneOf([
        iaa.Affine(rotate=(-15, 15), mode="edge"),
        iaa.Affine(rotate=(-30, 30), mode="edge"),
        iaa.Affine(rotate=(-45, 45), mode="edge"),
        iaa.Affine(rotate=(-60, 60), mode="edge"),
        iaa.Affine(rotate=(-75, 75), mode="edge"),
        iaa.Affine(rotate=(-90, 90), mode="edge"),
    ])
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug

def rotate(image, bbs, rotate_angle=(-90, 90)):
    aug = iaa.Affine(rotate=rotate_angle, mode="edge")
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def rotate2(image, bbs, rotate_angle=(-45, 45)):
    aug = iaa.Affine(rotate=rotate_angle, mode="edge")
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def shear(image, bbs, shear_angle=(-16, 16)):
    aug = iaa.Affine(shear=shear_angle, mode="edge")
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def translate(image, bbs):
    aug = iaa.OneOf([
        iaa.PerspectiveTransform(scale=(0.01, 0.15), mode="edge"),
        iaa.ElasticTransformation(alpha=(0, 2.0), sigma=0.1, mode="edge"),
        iaa.ScaleX((0.5, 1.5), mode="edge"),
        iaa.ScaleY((0.5, 1.5), mode="edge"),
        iaa.TranslateX(percent=(-0.1, 0.1), mode="edge"),
        iaa.TranslateY(percent=(-0.1, 0.1), mode="edge")
    ])
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug

def translate2(image, bbs):
    scale_t = 0.85
    aug = iaa.OneOf([
                                                                
        iaa.ScaleX((scale_t, 1.0/scale_t), mode="edge"),
        iaa.ScaleY((scale_t, 1.0/scale_t), mode="edge"),
        iaa.TranslateX(percent=(-0.1, 0.1), mode="edge"),
        iaa.TranslateY(percent=(-0.1, 0.1), mode="edge")
    ])
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def crop(image, bbs):
    aug = iaa.Sequential([
        iaa.CropAndPad(percent=(-0.2, 0.2), keep_size=True),
        iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))
    ])
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def oneof_aug(image, bbs):
    aug = iaa.OneOf([
                                       
        iaa.AdditiveGaussianNoise(scale=0.2 * 255),
        iaa.Add(50, per_channel=True),
        iaa.Sharpen(alpha=0.5)
    ])
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def resize(image, bbs, size=(1280, 1280)):
    image_rescaled = ia.imresize_single_image(image, size)
    bbs_rescaled = bbs.on(image_rescaled)
    return image_rescaled, bbs_rescaled


def blur(image, bbs):
    aug = iaa.OneOf([
        iaa.GaussianBlur(sigma=(0.0, 3.0)),
        iaa.AverageBlur(k=(2, 11)),
        iaa.AverageBlur(k=((5, 11), (1, 3))),
        iaa.BilateralBlur(
            d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),
        iaa.MedianBlur(k=(3, 11))
    ])
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


       
def clouds(image, bbs):
    aug = iaa.OneOf([
        iaa.Clouds(),
        iaa.Fog()
    ])
    image_aug, bbs_aug = aug(image=image, polygons=bbs)
    return image_aug, bbs_aug


def get_augs():
    augs = [seq, resize, oneof_aug, fliplr, flipud, flipone, rotate, rotate2, shear, crop, blur, translate]
    return augs


def xyxy2points(xyxy):
    return xyxy[0], xyxy[1], xyxy[2], xyxy[1], xyxy[2], xyxy[3], xyxy[0], xyxy[3]


def gen_img():
    img_path = r'/home/LIESMARS/2019286190105/datasets/final-master/HRSC/train/images'
    label_path = r'/home/LIESMARS/2019286190105/datasets/final-master/HRSC/train/labels'
    aug_img_path = r'../'
    aug_label_path = r'aug/aug_label'
    images = os.listdir(img_path)
    for img in images:
        src_image = cv2.imread(os.path.join(img_path, img))
        label = img.split('.')[0] + '.txt'
        polys = read_anno(label_path, label)
        bbs = PolygonsOnImage(polys, shape=src_image.shape)
        augs = get_augs()
        t = 1
        for aug in augs:
            image_aug, bbs_aug = aug(src_image, bbs)
                               
            bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
            aug_name = img.split('.')[0] + '-' + str(t)
            aug_label_name = 'aug_' + aug_name + '.txt'
            with open(os.path.join(aug_label_path, aug_label_name), 'w') as f:
                for i in range(len(bbs_aug.polygons)):
                    after = bbs_aug.polygons[i]
                    n = len(after.coords)
                    if n < 4 or n > 4:
                        continue
                    coords = after.coords
                    line = ""
                    for p in coords:
                        line += str(p[0]) + ' ' + str(p[1]) + ' '
                    line += after.label
                    f.write(line + '\n')
            aug_img_name = aug_name + '.jpg'
            cv2.imwrite(os.path.join(aug_img_path, aug_img_name), image_aug)
            t += 1

