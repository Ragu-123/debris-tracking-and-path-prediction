import cv2
import torch
import numpy as np

# Load YOLOv5 model (pre-trained or custom weights)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/SEC/runs/train/experiment2/weights/best.pt')

# Define the Kalman filter for continuous tracking
class KalmanFilter:
    def __init__(self):
        # Define Kalman filter
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # Lower noise for smooth tracking

    def predict(self, x, y):
        '''This function estimates the object's next position using the Kalman filter'''
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)
        prediction = self.kf.predict()
        return int(prediction[0]), int(prediction[1])  # Return predicted X, Y coordinates

# Instantiate Kalman Filter
kalman_filter = KalmanFilter()

# Define video input
video_path = "C:/Users/SEC/Downloads/ibm/sample video.mp4"
cap = cv2.VideoCapture(video_path)

# Video writer to save the output
output_video_path = "C:/Users/SEC/Downloads/ibm/output_with_kalman_continuous.mp4"
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))

# Store previous points to draw the path
previous_points = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model on the frame
    results = model(frame)

    # Get detected debris bounding boxes
    for *xyxy, conf, cls in results.xyxy[0]:
        x_min, y_min, x_max, y_max = map(int, xyxy)

        # Calculate center of the detected debris
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        # Predict the next position using Kalman Filter
        predicted_x, predicted_y = kalman_filter.predict(center_x, center_y)

        # Draw the detected debris bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Store the current predicted point
        previous_points.append((predicted_x, predicted_y))

        # Draw the continuous predicted path with lines
        for i in range(1, len(previous_points)):
            if previous_points[i - 1] is None or previous_points[i] is None:
                continue
            cv2.line(frame, previous_points[i - 1], previous_points[i], (0, 255, 0), 2)  # Draw predicted path (green line)

    # Save frame with predictions
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('Debris Tracking with Path Prediction', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
