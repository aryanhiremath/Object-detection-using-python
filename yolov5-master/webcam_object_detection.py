import cv2
import torch
from matplotlib import pyplot as plt

# Load the YOLO model (you can use pretrained models)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or 'yolov5m', 'yolov5l', etc. depending on the model you want to use

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

while True:
    ret, frame = cap.read()  # Read frame from webcam
    if not ret:
        print("Failed to grab frame")
        break

    # Run object detection
    results = model(frame)  # Perform inference on the webcam frame

    # Results
    results.render()  # Render detected objects on the frame

    # Display the result
    cv2.imshow("Object Detection", frame)

    # Exit the loop when the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()
