                                          
"""
Common modules
"""

import json
import math
import platform
import warnings
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, colorstr, increment_path, make_divisible,
                           non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import time_sync
import torch.nn.functional as F

def autopad(k, p=None):                   
                   
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]            
    return p


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
 
 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.register_buffer()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)


class Conv(nn.Module):
                          
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):                                                  
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))



                                    
                                              
                                                  
                                                 
                                                 

                                                                           
                               
                                                                           
    
                                           
                                                                               
                                                                    

                                     

                           
                                                                 
                                                                 
                                               
                    

 
                                    
                                        
                                                  

                                                                    
                                                

                                                                               
                                     

                           
                                                      
                                                        
                                                  
                          
                                
 
                        

                                                                           
                                      
                                            
                                                
                                        

                                                                                   
                                                     

                           
                                           
                                           
                  


class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
                                         
                                                        
                        
                                                        
           
        self.sigmoid = nn.Sigmoid()
                        
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
                                                   
                                                   
        return self.sigmoid(avgout + maxout)
        
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3) 
                        
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, c1,c2):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out



class CAM(nn.Module):
    def __init__(self, c1,c2):
        super(CAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(c1)

    def forward(self, x):
        out = self.channel_attention(x) * x
        return out



class SAM(nn.Module):
    def __init__(self, c1,c2):
        super(SAM, self).__init__()
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.spatial_attention(x) * x
        return out

class CSPCAM(nn.Module):
                                                     
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):                                                      
        super().__init__()
        c_ = int(c2 * e)                   
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)                 
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.attention = CAM(c2, c2)

    def forward(self, x):
        x = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        return self.attention(x)


class CSPSAM(nn.Module):
                                                     
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):                                                      
        super().__init__()
        c_ = int(c2 * e)                   
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)                 
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.attention = SAM(c2, c2)

    def forward(self, x):
        x = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        return self.attention(x)



class DWConv(Conv):
                                  
    def __init__(self, c1, c2, k=1, s=1, act=True):                                                  
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class TransformerLayer(nn.Module):
                                                                                                          
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
                                                         
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)                                
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
                         
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):                                              
        super().__init__()
        c_ = int(c2 * e)                   
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
                                                                            
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):                                                      
        super().__init__()
        c_ = int(c2 * e)                   
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)                            
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
                                        
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):                                                      
        super().__init__()
        c_ = int(c2 * e)                   
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)                 
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
                                                                                                

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class CSPC(nn.Module):
                                                      
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):                                                      
        super().__init__()
        c_ = int(c2 * e)                   
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)                 
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.attention = CBAM(c2, c2)

    def forward(self, x):
        x = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        return self.attention(x)



class C3TR(C3):
                                       
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
                          
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
                                      
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)                   
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
                                                                         
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2                   
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')                                             
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
                                                                            
    def __init__(self, c1, c2, k=5):                                   
        super().__init__()
        c_ = c1 // 2                   
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')                                             
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
                                       
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):                                                  
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
                                          

    def forward(self, x):                                 
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
                                            


class GhostConv(nn.Module):
                                                               
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):                                         
        super().__init__()
        c_ = c2 // 2                   
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
                                                              
    def __init__(self, c1, c2, k=3, s=1):                                 
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),      
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),      
                                  GhostConv(c_, c2, 1, 1, act=False))             
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
                                                                               
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()                                                            
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)                     
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()                     
        return x.view(b, c * s * s, h // s, w // s)                  


class Expand(nn.Module):
                                                                              
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()                                              
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)                     
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()                     
        return x.view(b, c // s ** 2, h * s, w * s)                   


class Concat(nn.Module):
                                                   
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class DetectMultiBackend(nn.Module):
                                                                        
    def __init__(self, weights='yolov5s.pt', device=None, dnn=True):
                
                                        
                                                    
                                             
                                                 
                                        
                                            
                                          
                                                        
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        suffix, suffixes = Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '', '.mlmodel']
        check_suffix(w, suffixes)                                        
        pt, onnx, tflite, pb, saved_model, coreml = (suffix == x for x in suffixes)                    
        jit = pt and 'torchscript' in w.lower()
        stride, names = 64, [f'class{i}' for i in range(1000)]                   

        if jit:               
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}                  
            model = torch.jit.load(w, _extra_files=extra_files)
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])                    
                stride, names = int(d['stride']), d['names']
        elif pt:           
            from models.experimental import attempt_load                                   
            model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
            stride = int(model.stride.max())                
            names = model.module.names if hasattr(model, 'module') else model.names                   
            try:
                mask_dir = model.module.mask_dir if hasattr(model, 'module') else model.mask_dir
            except Exception:
                mask_dir = [0] * len(names)
            pts = (model.module.get_module_byname('Detect2') if hasattr(model, 'module') else model.get_module_byname('Detect2')) is not None

        self.__dict__.update(locals())                                

    def forward(self, im, augment=False, visualize=False, val=False):
                                       
        b, ch, h, w = im.shape                                 
        if self.pt:           
            y = self.model(im) if self.jit else self.model(im, augment=augment, visualize=visualize)
                                       
            if val:
                out = []
                train_out = []
                for t in y:
                    out.append(t[0])
                    train_out.extend(t[1])
                return out, train_out
            return y
        y = torch.tensor(y)
        return (y, []) if val else y
    
    def get_module_byname(self, name:str):
        return self.model.module.get_module_byname(name) if hasattr(self.model, 'module') else self.model.get_module_byname(name)


class AutoShape(nn.Module):
                                                                                                                      
    conf = 0.25                            
    iou = 0.45                     
    classes = None                                                                                       
    multi_label = False                               
    max_det = 1000                                          

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def autoshape(self):
        LOGGER.info('AutoShape already enabled, skipping... ')                                                
        return self

    def _apply(self, fn):
                                                                                                          
        self = super()._apply(fn)
        m = self.model.get_module_byname('Detect')            
        m.stride = fn(m.stride)
        m.grid = list(map(fn, m.grid))
        if isinstance(m.anchor_grid, list):
            m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
                                                                                                    
                                                                           
                                                                          
                                                                                                
                                                                                               
                                                            
                                                                                                 
                                                                                                          

        t = [time_sync()]
        p = next(self.model.parameters())                       
        if isinstance(imgs, torch.Tensor):         
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)             

                     
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])                                    
        shape0, shape1, files = [], [], []                                         
        for i, im in enumerate(imgs):
            f = f'image{i}'            
            if isinstance(im, (str, Path)):                   
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):             
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:                
                im = im.transpose((1, 2, 0))                                          
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)                     
            s = im.shape[:2]       
            shape0.append(s)               
            g = (size / max(s))        
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)          
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]                   
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]       
        x = np.stack(x, 0) if n > 1 else x[0][None]         
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))                
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255                    
        t.append(time_sync())

        with amp.autocast(enabled=p.device.type != 'cpu'):
                       
            y = self.model(x, augment, profile)[0]           
            t.append(time_sync())

                          
            y = non_max_suppression(y, self.conf, iou_thres=self.iou, classes=self.classes,
                                    multi_label=self.multi_label, max_det=self.max_det)       
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
                                                   
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super().__init__()
        d = pred[0].device          
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]                  
        self.imgs = imgs                                  
        self.pred = pred                                               
        self.names = names               
        self.files = files                   
        self.xyxy = pred               
        self.xywh = [xyxy2xywh(x) for x in pred]               
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]                   
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]                   
        self.n = len(self.pred)                                 
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))                   
        self.s = shape                        

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '          
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()                        
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "                 
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):                           
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:              
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im           
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])        
            if save:
                f = self.files[i]
                im.save(save_dir / f)        
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)                 
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)                

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)                      
        self.display(save=True, save_dir=save_dir)                

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)                

    def render(self):
        self.display(render=True)                  
        return self.imgs

    def pandas(self):
                                                                                      
        new = copy(self)               
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'                
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'                
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]          
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
                                                                                     
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])                   
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
                                                        
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):                                                  
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)                  
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)                  
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)               
        return self.flat(self.conv(z))                      
