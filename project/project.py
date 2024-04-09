import cv2
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from sort import Sort  # Assuming you're using the SORT tracking algorithm
import numpy as np

# Load Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Initialize the tracker
tracker = Sort()

# Setup video input and output
video = cv2.VideoCapture('videos/input/mcgill_drive.mp4')
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('videos/output/out_mcgill_drive.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

COCO_CLASS_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def draw_boxes(frame, predictions):
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # Filter out detections with low confidence
            x1, y1, x2, y2 = box.int().tolist()
            class_name = COCO_CLASS_NAMES[label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return frame

def process_frame(frame, model):

    frame_tensor = [torch.tensor(frame.transpose((2, 0, 1))).float()]
   # frame_tensor = preprocess(frame)

    # Convert frame to tensor
    
    with torch.no_grad():
        prediction = model(frame_tensor)

    #draw_boxes(frame, prediction)
    #print(prediction)
    
    # Extract bounding boxes and scores for cars (class ID for car is 3 in COCO)
    # boxes = prediction[0]['boxes'][prediction[0]['labels'] == 3].cpu().numpy()
    # scores = prediction[0]['scores'][prediction[0]['labels'] == 3].cpu().numpy()

    boxes = prediction[0]['boxes']
    scores = prediction[0]['scores']

    print(boxes)
    
    # Filter detections with a confidence score above a threshold (e.g., 0.5)
    boxes = boxes[scores > 0.5]
    
    return boxes

unique_cars = 0
frame_count = 0

while True:
    ret, frame = video.read()
    if not ret or frame_count > 5:
        break
    print(f"Processing frame {frame_count}")
    frame_count += 1
    # Detect cars
    boxes = process_frame(frame, model)
    print(boxes)
    
    # Update tracker with new frame detections
    trackers = tracker.update(boxes)

    print(trackers)
    
    # Draw bounding boxes and display
    for d in trackers:
        d = d.astype(np.int32)
        cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), (255, 0, 0), 2)
        unique_cars = max(unique_cars, d[4])  # Update unique car count
    
    out.write(frame)

print(f"Total unique cars detected: {unique_cars}")

video.release()
out.release()
