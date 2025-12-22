from ultralytics import YOLO
import cv2
import json
import queue
import threading
import time
import redis

# -------------------------------
# Redis setup
# -------------------------------
# Redis connection details
REDIS_HOST = "localhost"   # Redis server running locally
REDIS_PORT = 6379
REDIS_CHANNEL = "yolo:detections"  # Channel used to publish YOLO results

# Create Redis client
# decode_responses=True ensures strings instead of bytes
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# -------------------------------
# Queue + worker for sending messages
# -------------------------------
# Queue decouples YOLO inference from Redis publishing
# This prevents blocking the main video loop if Redis is slow
payload_queue = queue.Queue()

def redis_worker():
    """
    Dedicated worker thread that publishes messages to Redis.
    Keeps network I/O out of the main inference loop.
    """
    while True:
        payload = payload_queue.get()

        # None is used as a shutdown signal
        if payload is None:
            break

        try:
            r.publish(REDIS_CHANNEL, json.dumps(payload))
        except Exception as e:
            print(f"Redis publish failed: {e}")

        payload_queue.task_done()

# Start Redis publisher thread
threading.Thread(target=redis_worker, daemon=True).start()

# -------------------------------
# YOLO model setup
# -------------------------------
# Load a pre-trained YOLO model
model = YOLO("yolo11n.pt")

# Alternative: fine-tuned custom model
# model = YOLO("models\\yolo11n-garbage.pt")

# -------------------------------
# Video stream setup
# -------------------------------
# IP camera stream URL
VIDEO_STREAM = "http://192.168.1.120:81/stream"

cap = cv2.VideoCapture(VIDEO_STREAM)

# Explicitly set frame resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"Stream setup done: {cap.isOpened()}")

# -------------------------------
# OFF detection tracking
# -------------------------------
# If no detections occur for this many seconds,
# an OFF state will be published to Redis
NO_DETECTION_TIMEOUT = 5  # seconds
last_detection_time = time.time()

def display_frame(annotated_frame, fps_avg):
    cv2.putText(
        annotated_frame,
        f"FPS: {fps_avg:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    cv2.imshow("YOLO Inference", annotated_frame)


# -------------------------------
# Camera stream processing loop
# -------------------------------
prev_frame_time = time.time()
try:
    while cap.isOpened():
        success, frame = cap.read()
        # Handle camera read failures gracefully
        if not success or frame is None:
            print("Failed to grab frame")
            time.sleep(0.1)
            continue

        # -------------------------------
        # YOLO inference
        # -------------------------------
        # Run object detection with confidence threshold
        results = model(frame, conf=0.5, verbose=False)
        current_time = time.time()

        # -------------------------------
        # Extract detections
        # -------------------------------
        detections = []

        for r_ in results:
            for cls, conf in zip(
                r_.boxes.cls.cpu().numpy(),
                r_.boxes.conf.cpu().numpy()
            ):
                label = r_.names[int(cls)]  # Convert class ID to label
                detections.append({
                    "label": label,
                    "confidence": float(conf)
                })

        # -------------------------------
        # Redis publishing logic
        # -------------------------------
        if len(detections) > 0:
            # Send detections when objects are found
            print(detections)
            payload_queue.put({"detections": detections})
            last_detection_time = current_time

        elif (current_time - last_detection_time) >= NO_DETECTION_TIMEOUT:
            # If no detections for a while, publish OFF state
            payload_queue.put({"status": "OFF"})
            last_detection_time = current_time
            print("No detections for timeout → OFF sent")


        # -------------------------------
        # FPS monitoring
        # -------------------------------
        now = time.time()
        dt = now - prev_frame_time
        prev_frame_time = now
        fps = 1.0 / max(dt, 1e-3)

        # -------------------------------
        # Visualization (optional)
        # -------------------------------
        # Draw bounding boxes and labels on the frame
        annotated_frame = results[0].plot()
        display_frame(annotated_frame, fps)
        # ESC key exits the loop
        if cv2.waitKey(1) & 0xFF == 27:
            break


finally:
    # -------------------------------
    # Cleanup
    # -------------------------------
    cap.release()
    cv2.destroyAllWindows()

    # Signal Redis worker to shut down
    payload_queue.put(None)
