# 9DPose1.0
9DPose is a real-time object detection and recognition model that outputs the 3D position, orientation, and physical dimensions (length, width, and height) of targets relative to the sensor. The model provides real-time 9-dimensional information for multiple targets simultaneously. Compared to conventional high-precision object detection models, 9DPose additionally estimates the target’s 3-axis orientation, physical size, spatial distance, and direction.
9DPose supports monocular vision-based real-time 3D object detection and recognition, making it suitable for aerial and ground target reconnaissance, identification, and measurement.
This 9DPose implementation incorporates several novel technologies, including quaternion-based pose regression, Rd-Dz virtual coordinate system transformation, and PKAttention. It is the first to achieve full three-degree-of-freedom (3-DoF) pose estimation for objects. Since it operates in a monocular, pure-vision mode without requiring specialized depth or ranging sensors, this technology is highly conducive to rapid deployment and widespread adoption.
Thank you for your interest!

9DPose Object Detection Demos
![Image description](demos/coco2017/000000013729.jpg)
![Image description](demos/coco2017/000000014226.jpg)
![Image description](demos/dota1.5/P0007_2_0.jpg)
![Image description](demos/dota1.5/P0128_8_0.jpg)
![Image description](demos/dota1.5-10terms/patches_P0000_84_0.jpg)
![Image description](demos/dota1.5-10terms/patches_P1067_1_0.jpg)


### Inference Visualization
Below is a video demonstration of the model inference on the  Plane9D and KITTI dataset:
<p align="center">
  <img src = "./demos/videos/9DPose.gif" width="80%">
  <img src = "./demos/videos/plane9D.gif" width="80%">
  <img src = "./demos/videos/kitti-yolo2.gif" width="80%">
</p>
<video width="640" height="360" controls>
  <source src="https://liujin1975060601.github.io/9dPose1.0/demos/videos/plane9D.mp4" type="video/mp4">
点击链接播放演示视频，请<a href="https://liujin1975060601.github.io/9dPose1.0/demos/videos/plane9D.mp4">点击这里播放视频</a>。
</video>
<video width="640" height="360" controls>
  <source src="https://liujin1975060601.github.io/9DPose1.0/demos/videos/plane9D.mp4" type="video/mp4">
点击链接播放演示视频，请<a href="https://liujin1975060601.github.io/yolov5-ft/demos/videos/road-cars-s_20250205_23160389_20250205_23205007.mp4">点击这里播放视频</a>。
</video>

The 9DPose model supports the following four datasets:
kitti
nuScene
Plane9D
NOSC

### Instructions
This code is an improvement based on the YOLOv5 architecture, with the training and validation operations and module organization consistent with YOLOv5.
- `train.py` starts training  
  (specify the training dataset `data`, model architecture `cfg`, and pre-trained weights `weights`).

- `val.py` starts validation  
  (specify model weights `weights` and dataset `data`).

- `detect.py` starts image or video detection  
  (specify model weights `weights` and the source folder path to be detected `source`).

