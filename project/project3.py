import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
import argparse
import sys
import matplotlib.pyplot as plt

def print_car_count(unique_cars, frame):
    cv2.putText(frame, f'Unique cars: {unique_cars}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return frame

def print_person_count(unique_persons, frame):
    cv2.putText(frame, f'Unique pedestrians: {unique_persons}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return frame

def process_video(
    source_weights_path: str,
    source_video_path: str,
    target_video_path: str,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> None:
    model = YOLO(source_weights_path)

    car_tracker = sv.ByteTrack()
    person_tracker = sv.ByteTrack()
    person_annotator = sv.RoundBoxAnnotator()
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)

    car_tracker_id_to_label = {}
    car_next_label = 1
    unique_cars = 0

    # structure to store each unique car's centroid position at every frame
    car_centroids = {}

    person_tracker_id_to_label = {}
    person_next_label = 1
    unique_persons = 0

    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            results = model(
                frame, verbose=False, conf=confidence_threshold, iou=iou_threshold
            )[0]
            detections = sv.Detections.from_ultralytics(results)


            #get only cars and persons
            car_detections = detections[detections.class_id == 2]
            car_detections = car_detections[car_detections.confidence > 0.7]
            car_detections = car_tracker.update_with_detections(car_detections)

            car_labels = []
            for tracker_id in car_detections.tracker_id:
                if tracker_id not in car_tracker_id_to_label:
                    car_tracker_id_to_label[tracker_id] = car_next_label
                    car_next_label += 1
                car_labels.append(f"Car {car_tracker_id_to_label[tracker_id]}")

            for detection in car_detections:
                coords, _, _, _, tracker_id, _ = detection
                x1, y1, x2, y2 = coords
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                if tracker_id not in car_centroids:
                    car_centroids[tracker_id] = []
                car_centroids[tracker_id].append(centroid)

            unique_cars = len(car_tracker_id_to_label)

            #print(labels)

            annotated_frame = box_annotator.annotate(
                scene=frame.copy(), detections=car_detections
            )

            annotated_labeled_frame = label_annotator.annotate(
                scene=annotated_frame, detections=car_detections, labels=car_labels
            )

            annotated_labeled_frame = print_car_count(unique_cars, annotated_labeled_frame)


            person_detections = detections[detections.class_id == 0]
            person_detections = person_detections[person_detections.confidence > 0.5]
            person_detections = person_tracker.update_with_detections(person_detections)

            person_labels = []
            for tracker_id in person_detections.tracker_id:
                if tracker_id not in person_tracker_id_to_label:
                    person_tracker_id_to_label[tracker_id] = person_next_label
                    person_next_label += 1
                person_labels.append(f"Person {person_tracker_id_to_label[tracker_id]}")

            unique_persons = len(person_tracker_id_to_label)

            annotated_frame = box_annotator.annotate(
                scene=annotated_labeled_frame.copy(), detections=person_detections
            )

            annotated_labeled_frame = label_annotator.annotate(
                scene=annotated_frame, detections=person_detections, labels=person_labels
            )

            annotated_labeled_frame = print_person_count(unique_persons, annotated_labeled_frame)

            sink.write_frame(frame=annotated_labeled_frame)

    print(f"Unique cars: {unique_cars}, Unique persons: {unique_persons}")

    for tracker_id in car_centroids:
        car_centroids[tracker_id] = np.array(car_centroids[tracker_id])

    #mean displacement vector of each car


    mean_displacement = np.zeros(len(car_centroids))
    for i, tracker_id in enumerate(car_centroids):
        if len(car_centroids[tracker_id]) < 2:
            continue
        mean_displacement[i] = np.mean(np.linalg.norm(np.diff(car_centroids[tracker_id], axis=0), axis=1))

    #replace nan values with 0
    mean_displacement = np.nan_to_num(mean_displacement)

    #print(mean_displacement)
    total_mean_displacement = np.mean(mean_displacement)
    std = np.std(mean_displacement)

    

    print(f"Mean displacement: {total_mean_displacement}, Standard deviation: {std}")


    #get number of values inside 1/2 standard deviation
    inside_std = np.sum(np.abs(mean_displacement - total_mean_displacement) < std/2)
    print(f"Estimated number of parked cars: {inside_std}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_to_process", type=int, help="Video to process. 1: mcgill_drive.mp4, 2: st-catherines_drive.mp4")

    args = parser.parse_args()

    if args.video_to_process == 1:
        video = "videos/input/mcgill_drive.mp4"
        output = "videos/output/out_mcgill_drive.mp4"
    elif args.video_to_process == 2:
        video = "videos/input/st-catherines_drive.mp4"
        output = "videos/output/out_st-catherines_drive.mp4"
    else:
        print("Invalid video. 1: mcgill_drive.mp4, 2: st-catherines_drive.mp4")
        sys.exit(1)
    
    process_video(
        source_weights_path="yolov8s.pt",
        source_video_path=video,
        target_video_path=output,
        confidence_threshold=0.5,
        iou_threshold=0.5,
    )
