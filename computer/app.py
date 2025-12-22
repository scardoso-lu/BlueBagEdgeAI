import cv2
import time
import json
import queue
import threading
import redis
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from ultralytics import YOLO
from PIL import Image, ImageTk

# -----------------------------
# Global stop signal
# -----------------------------
stop_event = threading.Event()

# -----------------------------
# App state
# -----------------------------
class AppState:
    def __init__(self):
        self.prev_frame_time = time.time()
        self.last_detection_time = time.time()
        self.prev_light_state = None
        self.same_light_state = 0

state = AppState()

# -----------------------------
# Redis setup
# -----------------------------
r = redis.Redis(host="localhost", port=6379, decode_responses=True)
payload_queue = queue.Queue()
light_queue = queue.Queue()

def redis_worker():
    while not stop_event.is_set():
        try:
            payload = payload_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if payload is None:
            break

        try:
            r.publish("yolo:detections", json.dumps(payload))
        except:
            pass

        payload_queue.task_done()

def redis_light_listener():
    pubsub = r.pubsub()
    pubsub.subscribe("yolo:light_state")

    try:
        for message in pubsub.listen():
            if stop_event.is_set():
                break
            if message["type"] == "message":
                light_queue.put(message["data"].strip().upper().replace('.', ''))
    finally:
        pubsub.close()

redis_pub_thread = threading.Thread(target=redis_worker, daemon=True)
redis_sub_thread = threading.Thread(target=redis_light_listener, daemon=True)

redis_pub_thread.start()
redis_sub_thread.start()

# -----------------------------
# YOLO model
# -----------------------------
model = YOLO("yolo11n.pt")
# model = YOLO("models/yolo11n-garbage.pt")

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture("http://192.168.1.120:81/stream")
CAM_SIZE = (640, 480)
NO_DETECTION_TIMEOUT = 8

# -----------------------------
# Shared frame buffer
# -----------------------------
latest_frame = None
frame_lock = threading.Lock()

# -----------------------------
# Logging helper
# -----------------------------
def log(msg):
    console.insert(tk.END, msg+"\n")
    console.see(tk.END)

# -----------------------------
# YOLO worker
# -----------------------------
YOLO_INTERVAL = 0.2 # ~20 FPS inference

def yolo_worker():
    global latest_frame

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, CAM_SIZE)

        results = model(frame, verbose=False)
        annotated = results[0].plot()

         # Extract detected object labels and confidence
        detections = [{"label": r_.names[int(cls)], "confidence": float(conf)}
        for r_ in results
            for cls, conf in zip(r_.boxes.cls.cpu().numpy(), r_.boxes.conf.cpu().numpy())]

        # -----------------------------
        # Publish detections to Redis
        # -----------------------------
        if detections:
            payload_queue.put({"detections": detections})  # Push detection data
            state.last_detection_time = time.time()  # Update last detection time
            log(json.dumps(detections))  # Log detections to console
        elif time.time() - state.last_detection_time >= NO_DETECTION_TIMEOUT:
            # If no detections for a while, send "OFF" status
            payload_queue.put({"status":"OFF"})
            state.last_detection_time = time.time()

        with frame_lock:
            latest_frame = annotated.copy()

    # SAFE PLACE TO RELEASE
    try:
        cap.release()
    except:
        pass


yolo_thread = threading.Thread(target=yolo_worker, daemon=True)
yolo_thread.start()

# -----------------------------
# Tkinter UI
# -----------------------------
root = tk.Tk()
root.title("YOLO Monitor")
root.geometry("1024x600")

top_frame = tk.Frame(root)
top_frame.pack(fill=tk.BOTH, expand=True)

video_label = tk.Label(top_frame, bg="black")
video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

light_frame = tk.Frame(top_frame, width=200, bg="gray")
light_frame.pack(side=tk.RIGHT, fill=tk.Y)

light_label = tk.Label(light_frame, text="OFF", font=("Arial", 36),
                       fg="white", bg="gray")
light_label.pack(expand=True, fill=tk.BOTH)

console = ScrolledText(root, height=8, bg="black", fg="lime",
                       font=("Consolas", 10))
console.pack(fill=tk.X)

def update_light_ui(state_value):
    color = {"TRUE": "green", "FALSE": "red"}.get(state_value, "gray")
    light_label.config(text=state_value, bg=color)

# -----------------------------
# UI update loop
# -----------------------------
def update_frame():
    if stop_event.is_set():
        return

    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None

    if frame is not None:
        now = time.time()
        fps = 1.0 / max(now - state.prev_frame_time, 1e-3)
        state.prev_frame_time = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    root.after(15, update_frame)

# -----------------------------
# Light message processing
# -----------------------------
def process_light_messages():
    if stop_event.is_set():
        return

    try:
        while True:
            msg = light_queue.get_nowait()
            print(msg)
            if msg == "OFF":
                update_light_ui("OFF")
                state.same_light_state = 0
            elif msg == state.prev_light_state:
                state.same_light_state += 1
            else:
                state.same_light_state = 0

            if state.same_light_state >= 2:
                update_light_ui(msg)
                state.same_light_state = 0

            state.prev_light_state = msg
    except queue.Empty:
        pass

    root.after(100, process_light_messages)

# -----------------------------
# Graceful shutdown
# -----------------------------
def on_close():
    stop_event.set()
    payload_queue.put(None)
    root.after(300, root.destroy)

root.protocol("WM_DELETE_WINDOW", on_close)

# -----------------------------
# Start loops
# -----------------------------
update_frame()
process_light_messages()
root.mainloop()
