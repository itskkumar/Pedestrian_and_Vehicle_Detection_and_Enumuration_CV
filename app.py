# Environment and library setup
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import gradio as gr
import cv2
import tempfile
import imageio
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Model and traker Initialization
model = YOLO("yolov8n.pt")

#Fuses model layers (e.g., Conv + BatchNorm) for speedup.
model.fuse()

#- Initializes DeepSORT with a track lifespan of 30 frames.
tracker = DeepSort(max_age=30)

# determined objects for track
TARGET_CLASSES = {"car", "motorcycle", "bus", "truck", "person"}

#draw custom lane polygone
LANE_POLYGON = np.array([[120, 250], [500, 250], [700, 384], [5, 384]], dtype=np.int32)

# lane Filter Function
def is_in_lane(x, y):
    return cv2.pointPolygonTest(LANE_POLYGON, (int(x), int(y)), False) >= 0

#main video processing function
def process_video(video_path):
    #object id collectors
    unique_vehicles = set()
    unique_lane_pedestrians = set()


    #video reading and writing setup
    try:
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data().get("fps", 25)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_out:
            writer = imageio.get_writer(temp_out.name, fps=fps)

            #per frame processing
            for frame in reader:
                frame = cv2.resize(frame, (640, 384))
                frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

                results = model(frame, verbose=False, conf=0.3, iou=0.5)[0]
                detections = []
#Detection structuring
                if results.boxes is not None:
                    for box in results.boxes:
                        try:
                            cls_id = int(box.cls.item())
                            label = model.names[cls_id]
                            if label not in TARGET_CLASSES:
                                continue
                            coords = box.xyxy[0]
                            if len(coords) != 4:
                                continue
                            x1, y1, x2, y2 = [int(c.item()) for c in coords]
                            w, h = x2 - x1, y2 - y1
                            conf = float(box.conf.item())
                            detections.append(([x1, y1, w, h], conf, label))
                        except Exception:
                            continue

                #track using deepsort
                #Prevent crash by only calling DeepSORT if detections exist
                tracks = tracker.update_tracks(detections, frame=frame) if detections else []

                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 0:
                        continue
                    track_id = track.track_id
                    label = track.get_det_class()
                    l, t, r, b = track.to_ltrb()
                    #object classification and annotation
                    cx, cy = (l + r) / 2, (t + b) / 2

                    if label == "person" and is_in_lane(cx, cy):
                        unique_lane_pedestrians.add(track_id)
                        color = (0, 255, 0)
                    elif label in {"car", "motorcycle", "bus", "truck"}:
                        unique_vehicles.add(track_id)
                        color = (0, 0, 255)
                    else:
                        continue

                    cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 2)
                    cv2.putText(frame, f"{label}-{track_id}", (int(l), int(t) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                #lane polygon overlay
                cv2.polylines(frame, [LANE_POLYGON], isClosed=True, color=(255, 255, 0), thickness=2)
                writer.append_data(frame)

            reader.close()
            writer.close()

        # final summary output
        summary = f" Vehicles: {len(unique_vehicles)} | Pedestrians in lane: {len(unique_lane_pedestrians)}"
        return temp_out.name, summary

    except Exception as e:
        return video_path, f" Error during processing: {str(e)}"

#Gradio interface setup 

demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Traffic Video"),
    outputs=[
        gr.Video(label="Annotated Output"),
        gr.Textbox(label="Tracking Summary")
    ],
    title="Lane-Based Tracker (YOLOv8 + DeepSORT)",
    description="Draws yellow polygon as a lane. Tracks people and vehicles within that region.",
    theme="soft",
    examples=[
        ["video1.mp4"],
        ["video2.mp4"],
        ["video3.mp4"]
    ],
    cache_examples=False  # Prevent crash from pre-caching
)

#launch app
if __name__ == "__main__":
    demo.launch(share=False, debug=True)
