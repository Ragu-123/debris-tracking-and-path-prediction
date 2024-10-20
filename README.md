# debris-tracking-using-yolov5-and-path-prediction

This project demonstrates a space debris tracking system using YOLOv5 for object detection and a Kalman Filter to predict and track the debris's trajectory in video footage.

## Overview
The goal of this project is to detect debris in video frames using a YOLOv5 model, and then continuously track and predict the future positions of the detected debris using a Kalman Filter. The output video shows both the detected debris and the predicted path.

## Features
Real-Time Detection: YOLOv5 detects debris in video frames.
Kalman Filter Prediction: Predicts the debris's future position based on previous frame positions.
Visual Tracking: Draws bounding boxes and predicted trajectory on each frame.
Video Output: Saves a new video file with tracked debris and predictions.

## Prerequisites
Python 3.7+
OpenCV
PyTorch

## output

### detected image:
![image](https://github.com/user-attachments/assets/1458de5d-242e-44a0-9805-abc9a176970a)

### output video
https://github.com/user-attachments/assets/3e578780-6b00-4f8c-8153-2e08f503a43d

