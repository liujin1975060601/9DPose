                                          
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import imp
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]                         
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))                    
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))            

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, scale_coords_poly,strip_optimizer, xyxy2xywh, rot_nms)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from models.experimental import attempt_load
from tools.plotbox import plot_one_rot_box

from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first
from utils.general import (LOGGER, check_dataset, check_file, check_git_status, check_img_size, check_requirements,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods, one_cycle,
                           print_args, print_mutation, strip_optimizer)

                                                                                                                                
def detect(model, training, im,augment,conf_thres, iou_thres, ab_thres, fold_angle,mask_dir=[],threshs=torch.zeros(0),dt=[]):
               
    if dt!=[]:
                   
        t2 = time_sync()
        if training:
            out,train_out = [],[]
            tmp = model(im, augment=augment)
            for t in tmp:
                out.append(t[0])
                train_out.extend(t[1])
        else:
            out, train_out = model(im, augment=augment, val=True)
        t3 = time_sync()
        dt[1] += t3 - t2          

                                 
                                                                                    
        out = rot_nms(out, conf_thres, iou_thres, ab_thres, fold_angle, mask_dir,threshs)
                                         
        dt[2] += time_sync() - t3           
        
        return out,train_out
    else:
        pred = model(im, augment=0, val=True)[0]
                                                                                                      
        pred = rot_nms(pred, conf_thres, iou_thres, ab_thres=ab_thres, fold_angle=fold_angle, mask_dir=mask_dir,threshs=threshs)
                                               
        return pred

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',                    
        source=ROOT / 'data/images',                                   
        imgsz=640,                           
        conf_thres=0.1,                        
        ab_thres=3,
        iou_thres=0.45,                     
        device='',                                         
        save_txt=False,                         
        nosave=False,                             
        classes=None,                                                
        agnostic_nms=False,                      
        augment=False,                       
        visualize=False,                      
        update=False,                     
        project=ROOT / 'runs/detect',                                
        name='exp',                                
        exist_ok=False,                                              
        line_thickness=1,                                   
        half=True,                                     
        plot_label=True,
        dir_line=False,
        fold=2,
        data='',
        nc=8,
        mask_dir=[]
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')                         
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)            
    else:
        if not os.path.exists(source):
            print(f'\033[91mimages_path:{source} not exist.\033[0m')

                 
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)                 
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)            

                
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False)
                                                                   
                                                                                               
    stride, names, mask_dir = model.stride, model.names, mask_dir or model.mask_dir

                                                                 

          
    pt = True
    half = True
    if pt:
        model.model.half() if half else model.model.float()
                
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True                                                      
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=False)
        bs = len(dataset)              
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=False)
        bs = 1              
    
    train_path = Path(weights).parent.parent
    if(os.path.exists(train_path / 'threshs.npy')):
        threshs = np.load(train_path / 'threshs.npy')
        iou_thres = min(threshs)
    else:
        threshs = torch.zeros(0)
        
                   
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))          
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        print(path)
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()                    
        im /= 255                        
        if len(im.shape) == 3:
            im = im[None]                        
        t2 = time_sync()
        dt[0] += t2 - t1

                   
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = detect(model, False, im,augment,conf_thres, iou_thres, ab_thres, fold_angle=fold, mask_dir=mask_dir,threshs=threshs)
        t3 = time_sync()
        dt[1] += t3 - t2
        dt[2] += t3 - t2
        '''
        pred = model(im, augment=augment, visualize=visualize, val=True)[0]
        #pred[2=Detect+Detect2][nl][b,a,H,W,Detect(5+c) or Detect2(5=1+2(dir)+2(ab))]
        t3 = time_sync()
        dt[1] += t3 - t2
        # NMS
        pred = rot_nms(pred, conf_thres, iou_thres, ab_thres=ab_thres, fold_angle=fold,mask_dir=mask_dir, threshs=threshs)
        #pred[b][nt,10=2*4+1(conf)+1(cls)]
        dt[2] += time_sync() - t3
        '''

        colors = [
        (54, 67, 244),
                        
        (176, 39, 156),
        (183, 58, 103),
        (181, 81, 63),
        (243, 150, 33),
        (212, 188, 0),
        (136, 150, 0),
        (80, 175, 76),
        (74, 195, 139),
        (57, 220, 205),
        (59, 235, 255),
        (0, 152, 255),
        (34, 87, 255),
        (72, 85, 121),
        (180, 105, 255)]

                             
        for i, det in enumerate(pred):                     
            seen += 1
            if webcam:                   
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)           
            save_path = str(save_dir / p.name)          
            txt_path = str(save_dir / 'labels' / p.stem)
            if not os.path.exists(str(save_dir / 'labels')):
                os.makedirs(str(save_dir / 'labels'))
            s += '%gx%g ' % im.shape[2:]                
            if len(det):
                                                         
                det[:, :8] = scale_coords_poly(im.shape[2:], det[:, :8], im0.shape).round()

                               
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()                        
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "                 

                               
                for *xyxy, conf, cls in reversed(det):
                    line = (cls, *xyxy, conf)               
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    xyxy = [x.cpu() for x in xyxy]
                    label = '{} {:.2f}'.format(names[int(cls)], conf)
                    if not plot_label:
                        label = None
                    plot_one_rot_box(np.array(xyxy), im0, color=colors[int(cls)%len(colors)], label=label, dir_line=dir_line, line_thickness=line_thickness)
                cv2.imwrite(save_path, im0)
            else:
                with open(txt_path+'.txt', 'w') as f:
                    pass
                                         
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
    total_time = dt[1]
    n_img = len(dataset)
    print(f"total inference time: {total_time}, \
        every inference time:{total_time / n_img}, fps:{1/(total_time / n_img)}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--ab_thres', type=float, default=3.0, help='a b thres')
    parser.add_argument('--plot_label',action='store_true', help='plot labels')
    parser.add_argument('--fold',type=int, default=1, help='fold angle')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1          
                                       
                                                           
    opt.data = 'data/Guge.yaml'
    opt.weights = 'runs/train/exp_Guge2/weights/best.pt'
    opt.source = '/home/liu/data/home/liu/workspace/darknet/datas/GuGe/val/images'
    opt.conf_thres = 0.3
    opt.iou_thres = 0.1
    opt.imgsz = [768, 768]
    opt.ab_thres = 7.0
    opt.fold = 2

    opt.name += f'_{Path(opt.data).stem}'
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
     
    LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(opt.data)                 
    opt.nc = int(data_dict['nc'])                     
    opt.mask_dir = data_dict.get('mask_dir', None)
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
