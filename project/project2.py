import cv2
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Load a pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# COCO dataset class names (index 0 is the background)
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

# Function to draw bounding boxes and class labels on a frame
def draw_boxes(frame, predictions):
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    for box, label, score in zip(boxes, labels, scores):
        if (label == 3 or label == 1) and score > 0.7:  # Filter out detections with low confidence
            x1, y1, x2, y2 = box.int().tolist()
            class_name = COCO_CLASS_NAMES[label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return frame

# Setup video reader and writer
video = cv2.VideoCapture('videos/input/mcgill_drive.mp4')
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
output = cv2.VideoWriter('videos/output/out_mcgill_drive.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


frame_count = 0
while True:
    ret, frame = video.read()

    if not ret:
        break
    print(f'Processing frame {frame_count}')
    frame_count += 1
    # Convert the frame to a format suitable for the model
    image = F.to_tensor(frame).unsqueeze_(0)
    
    with torch.no_grad():
        prediction = model(image)
    
    # Annotate frame
    annotated_frame = draw_boxes(frame, prediction)
    
    # Write the annotated frame to the output video
    output.write(annotated_frame)

# Cleanup
video.release()
output.release()
