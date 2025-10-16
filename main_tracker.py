import cv2
import time
import sys
import torch
import numpy as np
from ultralytics import YOLO
import norfair
from threading import Thread
from queue import Queue
from collections import deque

# --- تنظیمات اصلی ---
VIDEO_PATH = 'f.mp4'
SHOW_FRAME_DELAY_MS = 1 # یا 10 برای نمایش روان‌تر

# --- تنظیمات YOLO ---
YOLO_MODEL_PATH = 'yolov8n.pt'
CONF_THRESHOLD = 0.3 # آستانه YOLO
ALLOWED_CLASSES = [2, 3, 5, 7]
MIN_BOX_WIDTH = 20
MIN_BOX_HEIGHT = 20

# --- Norfair ---
def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points[0] - tracked_object.estimate[0])

DISTANCE_THRESHOLD = 30
HIT_COUNTER_MAX = 5 # ترک باید در 5 فریم دیده شود تا تایید گردد (می‌توانید 3 کنید)

# --- تخمین سرعت ---
CALIBRATED_DISTANCE = 15.0
TARGET_HEIGHT = 270
TARGET_WIDTH = 480
line1_y = int(0.34 * TARGET_HEIGHT)
line2_y = int(0.66 * TARGET_HEIGHT)
MIN_DT_FRAMES = 3
MAX_SPEED = 180.0
BOX_BUFFER_SIZE = 5

# --- DETECTION_INTERVAL = 1 ---
DETECTION_INTERVAL = 1 # <<< اجرای تشخیص در هر فریم >>>

# --- تابع خواندن فریم ---
def frame_reader(cap, queue, target_size):
    # ... (بدون تغییر) ...
    print("Frame reader thread started.")
    while True:
        ret, frame = cap.read()
        if not ret: queue.put(None); print("Frame reader: End."); break
        try:
            frame_resized = cv2.resize(frame, target_size)
            queue.put(frame_resized)
        except Exception as e: print(f"Frame reader error: {e}"); queue.put(None); break
    print("Frame reader thread finished.")


# --- تابع اصلی برنامه ---
def run_norfair_tracking():
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): print(f"Error opening video: {VIDEO_PATH}"); return

    fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fps_used = 30.0 if (fps is None or fps <= 5 or fps > 120) else fps
    frame_time = 1.0 / actual_fps_used
    if actual_fps_used == 30.0: print(f"Warning: Using default 30 FPS.")
    print(f"Video FPS (nominal): {actual_fps_used:.2f}, Frame time: {frame_time:.4f}s")
    print(f"DETECTION INTERVAL: {DETECTION_INTERVAL}")

    print(f"Loading YOLO: {YOLO_MODEL_PATH}")
    try: model = YOLO(YOLO_MODEL_PATH)
    except Exception as e: print(f"Error loading YOLO: {e}"); cap.release(); return

    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    try: model.to(device)
    except Exception as e: print(f"Error setting YOLO device: {e}"); cap.release(); return

    tracker = norfair.Tracker(
        distance_function=euclidean_distance,
        distance_threshold=DISTANCE_THRESHOLD,
        hit_counter_max=HIT_COUNTER_MAX,
    )
    print(f"Norfair tracker initialized. DistThresh={DISTANCE_THRESHOLD}, HitCounter={HIT_COUNTER_MAX}")

    track_history = {}
    frame_queue = Queue(maxsize=max(10, int(actual_fps_used)))
    target_size = (TARGET_WIDTH, TARGET_HEIGHT)
    reader_thread = Thread(target=frame_reader, args=(cap, frame_queue, target_size))
    reader_thread.daemon = True
    reader_thread.start()

    print("Starting processing loop...")
    processing_start_time = time.time(); processed_frames_count = 0; frame_index = 0

    while True:
        frame = frame_queue.get();
        if frame is None: break
        frame_index += 1; processed_frames_count += 1
        display_frame = frame.copy()

        # --- YOLO Detection (هر فریم) ---
        norfair_detections = []
        try:
            results = model(display_frame, conf=CONF_THRESHOLD, classes=ALLOWED_CLASSES, verbose=False)[0]
            for box in results.boxes:
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = coords
                conf = float(box.conf[0])
                if (x2 - x1) < MIN_BOX_WIDTH or (y2 - y1) < MIN_BOX_HEIGHT: continue
                cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
                detection_data = {'bbox': [x1, y1, x2, y2], 'conf': conf}
                norfair_detections.append(norfair.Detection(points=np.array([[cx, cy]]), data=detection_data))
        except Exception as e: print(f"Error YOLO inference: {e}")

        # --- Norfair Update ---
        try:
            tracked_objects = tracker.update(detections=norfair_detections)
        except Exception as e: print(f"Error Norfair update: {e}"); tracked_objects = []

        # --- پردازش و رسم ترک‌ها ---
        for tracked_object in tracked_objects:
            # <<< فیلتر تایید فعال است >>>
            if tracked_object.hit_counter < HIT_COUNTER_MAX:
                continue # رد کردن ترک‌های تایید نشده

            if tracked_object.last_detection.data is None: continue
            x1, y1, x2, y2 = map(int, tracked_object.last_detection.data['bbox'])
            track_id = tracked_object.id

            # --- محاسبه سرعت ---
            estimated_center = tracked_object.estimate[0]; current_cy = estimated_center[1]
            if track_id not in track_history: track_history[track_id] = {'line1_frame': None, 'line2_frame': None, 'speed_kph': None, 'last_seen': frame_index, 'prev_cy': current_cy, 'box_buffer': deque(maxlen=BOX_BUFFER_SIZE)}
            else: track_history[track_id]['last_seen'] = frame_index
            prev_cy = track_history[track_id].get('prev_cy', current_cy)
            crossed_line1 = prev_cy < line1_y <= current_cy
            crossed_line2 = prev_cy < line2_y <= current_cy
            if track_history[track_id]['line1_frame'] is None and crossed_line1: track_history[track_id]['line1_frame'] = frame_index
            if (track_history[track_id]['line1_frame'] is not None and track_history[track_id]['line2_frame'] is None and crossed_line2):
                 track_history[track_id]['line2_frame'] = frame_index
                 dt_frames = track_history[track_id]['line2_frame'] - track_history[track_id]['line1_frame']
                 if dt_frames >= MIN_DT_FRAMES:
                     time_elapsed = dt_frames * frame_time
                     if time_elapsed > 0.001:
                         speed_mps = CALIBRATED_DISTANCE / time_elapsed
                         speed_kph = speed_mps * 3.6
                         if 0 < speed_kph <= MAX_SPEED: track_history[track_id]['speed_kph'] = speed_kph
                         elif speed_kph > MAX_SPEED: track_history[track_id]['speed_kph'] = MAX_SPEED
            track_history[track_id]['prev_cy'] = current_cy

            # --- هموارسازی و رسم ---
            track_history[track_id]['box_buffer'].append([x1, y1, x2, y2])
            valid_boxes = [b for b in track_history[track_id]['box_buffer'] if b is not None]
            if len(valid_boxes) > 0:
                 smooth_box = np.mean(valid_boxes, axis=0).astype(int); sx1, sy1, sx2, sy2 = smooth_box
                 if sx1 >= sx2 or sy1 >= sy2: sx1, sy1, sx2, sy2 = x1, y1, x2, y2
            else: sx1, sy1, sx2, sy2 = x1, y1, x2, y2

            color = (0, 255, 0) # سبز (چون فیلتر تایید فعال است)
            cv2.rectangle(display_frame, (sx1, sy1), (sx2, sy2), color, 2)
            label = f"ID:{track_id}"
            speed_info = track_history[track_id].get('speed_kph')
            if speed_info is not None: label += f" {speed_info:.1f}km/h"
            cv2.putText(display_frame, label, (sx1, sy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- رسم خطوط و FPS ---
        cv2.line(display_frame, (0, line1_y), (display_frame.shape[1], line1_y), (0, 255, 255), 1)
        cv2.line(display_frame, (0, line2_y), (display_frame.shape[1], line2_y), (0, 255, 255), 1)
        processing_time_total = time.time() - processing_start_time
        avg_fps_processing = processed_frames_count / processing_time_total if processing_time_total > 0 else 0
        fps_text = f"Proc FPS: {avg_fps_processing:.1f}"
        cv2.putText(display_frame, fps_text, (10, TARGET_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- نمایش فریم ---
        cv2.imshow("Norfair Tracking (Stable Interval=1)", display_frame)
        if cv2.waitKey(SHOW_FRAME_DELAY_MS) & 0xFF == 27: break

    # --- پایان ---
    print("Releasing resources...")
    cap.release(); cv2.destroyAllWindows()
    if reader_thread.is_alive(): print("Waiting for frame reader thread...")
    if processing_start_time and processed_frames_count > 0:
        total_time = time.time() - processing_start_time
        if total_time > 0:
             print(f"Finished. Processed {processed_frames_count} frames in {total_time:.2f}s.")
             print(f"Average Tracking FPS: {processed_frames_count / total_time:.2f}")

# --- اجرا ---
if __name__ == "__main__":
    try: run_norfair_tracking()
    except Exception as e: print(f"An error occurred: {e}"); import traceback; traceback.print_exc()
    finally: cv2.destroyAllWindows(); print("Exiting program."); sys.exit(0)
