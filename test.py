import torch
from PIL import Image
import cv2

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/SEC/runs/train/experiment2/weights/best.pt')

# Load an image or video to test
image_path = 'C:/Users/SEC/Downloads/ibm/debris-detection/test/8.jpg'
img = Image.open(image_path)

# Perform inference
results = model(img)

# Display results
results.show()
