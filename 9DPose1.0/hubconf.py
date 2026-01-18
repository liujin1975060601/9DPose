                                          
"""
PyTorch Hub models https://pytorch.org/hub/ultralytics_yolov5/

Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
"""

import torch


def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    """Creates a specified YOLOv5 model

    Arguments:
        name (str): name of model, i.e. 'yolov5s'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model
        verbose (bool): print all information to screen
        device (str, torch.device, None): device to use for model parameters

    Returns:
        YOLOv5 pytorch model
    """
    from pathlib import Path

    from models.experimental import attempt_load
    from models.yolo import Model
    from utils.downloads import attempt_download
    from utils.general import check_requirements, intersect_dicts, set_logging
    from utils.torch_utils import select_device

    file = Path(__file__).resolve()
    check_requirements(exclude=('tensorboard', 'thop', 'opencv-python'))
    set_logging(verbose=verbose)

    save_dir = Path('') if str(name).endswith('.pt') else file.parent
    path = (save_dir / name).with_suffix('.pt')                   
    try:
        device = select_device(('0' if torch.cuda.is_available() else 'cpu') if device is None else device)

        if pretrained and channels == 3 and classes == 80:
            model = attempt_load(path, map_location=device)                            
        else:
            cfg = list((Path(__file__).parent / 'models').rglob(f'{name}.yaml'))[0]                   
            model = Model(cfg, channels, classes)                
            if pretrained:
                ckpt = torch.load(attempt_download(path), map_location=device)        
                csd = ckpt['model'].float().state_dict()                                 
                csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])             
                model.load_state_dict(csd, strict=False)        
                if len(ckpt['model'].names) == classes:
                    model.names = ckpt['model'].names                             
        if autoshape:
            model = model.autoshape()                                          
        return model.to(device)

    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = 'Cache may be out of date, try `force_reload=True`. See %s for help.' % help_url
        raise Exception(s) from e


def custom(path='path/to/model.pt', autoshape=True, verbose=True, device=None):
                                  
    return _create(path, autoshape=autoshape, verbose=verbose, device=device)


def yolov5n(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
                                                             
    return _create('yolov5n', pretrained, channels, classes, autoshape, verbose, device)


def yolov5s(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
                                                              
    return _create('yolov5s', pretrained, channels, classes, autoshape, verbose, device)


def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
                                                               
    return _create('yolov5m', pretrained, channels, classes, autoshape, verbose, device)


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
                                                              
    return _create('yolov5l', pretrained, channels, classes, autoshape, verbose, device)


def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
                                                               
    return _create('yolov5x', pretrained, channels, classes, autoshape, verbose, device)


def yolov5n6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
                                                                
    return _create('yolov5n6', pretrained, channels, classes, autoshape, verbose, device)


def yolov5s6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
                                                                 
    return _create('yolov5s6', pretrained, channels, classes, autoshape, verbose, device)


def yolov5m6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
                                                                  
    return _create('yolov5m6', pretrained, channels, classes, autoshape, verbose, device)


def yolov5l6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
                                                                 
    return _create('yolov5l6', pretrained, channels, classes, autoshape, verbose, device)


def yolov5x6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
                                                                  
    return _create('yolov5x6', pretrained, channels, classes, autoshape, verbose, device)


if __name__ == '__main__':
    model = _create(name='yolov5s', pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)              
                                                       

                      
    from pathlib import Path

    import cv2
    import numpy as np
    from PIL import Image

    imgs = ['data/images/zidane.jpg',            
            Path('data/images/zidane.jpg'),        
            'https://ultralytics.com/images/zidane.jpg',       
            cv2.imread('data/images/bus.jpg')[:, :, ::-1],          
            Image.open('data/images/bus.jpg'),       
            np.zeros((320, 640, 3))]         

    results = model(imgs)                     
    results.print()
    results.save()
