# Space-debris-tracking-using-yolov5-and-path-prediction

This project demonstrates a space debris tracking system using YOLOv5 for object detection and a Kalman Filter to predict and track the debris's trajectory in real time.

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

![confusion_matrix](https://github.com/user-attachments/assets/2c8f8ea3-2b2e-4530-98d4-6e6feb93a85f)


### detected image:
![image](https://github.com/user-attachments/assets/1458de5d-242e-44a0-9805-abc9a176970a)
![path tracked image](https://github.com/user-attachments/assets/94eb5547-ef0d-4439-90c9-e8c8ff075ee5)


### output video
https://github.com/user-attachments/assets/e1a1c3eb-fc12-4af9-bb72-9d2697f58d7e



