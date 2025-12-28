import degirum as dg
import degirum_tools as dgt
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import os
import requests
import redis
import io
# ============================
# Configuration
# ============================
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 450
UI_REFRESH_MS = 1  

MODELS = {
    "Face": "yolov8n_relu6_face--640x640_quant_hailort_multidevice_1",
    "Detection": "yolo11n_coco--640x640_quant_hailort_multidevice_1",
    "Segmentation": "yolov8n_relu6_coco_seg--640x640_quant_hailort_hailo8l_1",
    "Pose": "yolov8n_relu6_coco_pose--640x640_quant_hailort_hailo8l_1",
}

# Default sources dictionary: {label: url_or_path}
DEFAULT_VIDEOS = {
    "Traffic": "https://github.com/DeGirum/PySDKExamples/raw/refs/heads/main/images/Traffic2.mp4",
    "Hand Palm": "https://github.com/DeGirum/PySDKExamples/raw/refs/heads/main/images/HandPalm.mp4",
    "Walking People ": "https://github.com/DeGirum/PySDKExamples/raw/refs/heads/main/images/WalkingPerson.mp4",
    "Aerial Pedestrians": "https://github.com/DeGirum/PySDKExamples/raw/refs/heads/main/images/aerial_crossing_pedestrians_bikes_cropped.mp4",
    "Horses Mountain": "https://github.com/DeGirum/PySDKExamples/raw/refs/heads/main/images/horses_mountain_pasture.mp4",
    "Person Pose": "https://github.com/DeGirum/PySDKExamples/raw/refs/heads/main/images/person_pose.mp4",
    "Store": "https://github.com/DeGirum/PySDKExamples/raw/refs/heads/main/images/store.mp4",
}

DEFAULT_CAMERA_INDEX = "0"


VIDEO_FOLDER = "./videos"  # local folder to save downloaded videos
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Redis configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PUB_CHANNEL = "inference_frames"
REDIS_SUB_CHANNEL = "commands"

# ============================
# Model Loader
# ============================

def load_dg_model(model_name):
    model = dg.load_model(
        model_name=model_name,
        inference_host_address="@local",
        zoo_url="degirum/hailo",
    )

    model.overlay_show_labels = True
    model.overlay_show_confidence = True
    model.overlay_show_bbox = True
    model.confidence_threshold = 0.4

    return model

def get_local_video(label_or_url):
    """
    Given a label (from DEFAULT_VIDEOS) or a direct URL/path:
    - Checks if video exists locally
    - Downloads it if needed
    Returns the local path to the video
    """
    # Determine URL
    if label_or_url in DEFAULT_VIDEOS:
        url = DEFAULT_VIDEOS[label_or_url]
        filename = os.path.basename(url)
    else:
        url = label_or_url
        filename = os.path.basename(url)
    
    local_path = os.path.join(VIDEO_FOLDER, filename)
    
    # If already exists, return path
    if os.path.exists(local_path):
        return local_path
    
    # If URL starts with http, download it
    if url.startswith("http"):
        print(f"Downloading {url} -> {local_path}")
        try:
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete")
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None
    else:
        # Local path, but file does not exist
        print(f"Local video not found: {url}")
        return None

    return local_path

# ============================
# Tkinter App
# ============================
class InferenceUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
        self.redis_sub = self.redis_client.pubsub()
        self.redis_sub.subscribe(REDIS_SUB_CHANNEL)
        self.redis_thread = threading.Thread(target=self.redis_listener, daemon=True)
        self.redis_thread.start()

        self.title("Hailo-8L Edge AI Inference")
        self.geometry(f"1024x600")  # Width for controls + video

        self.running = False
        self.count_console = 0
        self.worker_thread = None
        self.model = None

        self.model_choice = tk.StringVar(value="Detection")
        self.camera_url = tk.StringVar(value="http://192.168.1.145:8080/video")

        self.source_type = tk.StringVar(value="Video File")
        self.source_value = tk.StringVar(value=list(DEFAULT_VIDEOS.keys())[0])

        self._build_layout()
        self.reload_model()

    # ------------------------
    # Layout
    # ------------------------
    def _build_layout(self):
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=3)

        # Left panel
        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="nsw")

        ttk.Label(left, text="Parameters", font=("Arial", 14, "bold")).pack(anchor="w", pady=10)
        ttk.Label(left, text="Model").pack(anchor="w")
        ttk.OptionMenu(left, self.model_choice, self.model_choice.get(), *MODELS.keys()).pack(fill="x")
        ttk.Button(left, text="Load Model", command=self.reload_model).pack(fill="x", pady=5)

        ttk.Label(left, text="Video Source").pack(anchor="w", pady=(15, 0))
        ttk.OptionMenu(
            left,
            self.source_type,
            self.source_type.get(),
            "Camera",
            "HTTP",
            "Video File",
            command=self.on_source_change
        ).pack(fill="x")
        # New (dropdown for default videos)
        self.video_dropdown_frame = ttk.Frame(left)
        self.video_dropdown_frame.pack(fill="x", pady=5)

        self.video_dropdown = ttk.OptionMenu(
            self.video_dropdown_frame,
            self.source_value,
            list(DEFAULT_VIDEOS.keys())[0],  # default selection
            *DEFAULT_VIDEOS.keys()
        )
        self.video_dropdown.pack(fill="x")
        ttk.Label(left, text="URL").pack(anchor="w", pady=(15, 0))
        ttk.Entry(left, textvariable=self.camera_url).pack(fill="x")

        ttk.Button(left, text="Start", command=self.start).pack(fill="x", pady=10)
        ttk.Button(left, text="Stop", command=self.stop).pack(fill="x")

        # Loading indicator
        self.loading_label = ttk.Label(left, text="", foreground="blue")
        self.loading_label.pack(pady=(5,0))

        # Right panel
        right = ttk.Frame(self, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)  # only video expands

        # Console (top)
        self.console = tk.Text(right, height=4, bg="black", fg="lime")
        self.console.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        # Video frame (middle, fixed size)
        video_frame = ttk.Frame(right, width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
        video_frame.grid(row=1, column=0, sticky="nsew")
        video_frame.grid_propagate(False)  # prevent resizing to children
        self.video_label = tk.Label(video_frame, bg="gray")
        self.video_label.place(x=0, y=0, width=VIDEO_WIDTH, height=VIDEO_HEIGHT)

        # Initially hide dropdown if source is not Video File
        self.on_source_change()

    # ------------------------
    # Source Handling
    # ------------------------
    def on_source_change(self, *_):
        self.stop()
        if self.source_type.get() == "Camera":
            self.source_value.set(DEFAULT_CAMERA_INDEX)
            self.video_dropdown.config(state="disabled")
        elif self.source_type.get() == "HTTP":
            self.source_value.set(DEFAULT_HTTP_URL)
            self.video_dropdown.config(state="disabled")
        else:  # Video File
            self.source_value.set(list(DEFAULT_VIDEOS.keys())[0])
            self.video_dropdown.config(state="normal")

        self.video_dropdown.update_idletasks()


    # ------------------------
    # Model Control
    # ------------------------
    def reload_model(self):
        self.stop()

        self.console.insert(tk.END, "Loading model...\n")
        self.update_idletasks()

        self.model = load_dg_model(MODELS[self.model_choice.get()])
        self.console.insert(tk.END, "Model ready\n")



    # ------------------------
    # Threaded Control
    # ------------------------
    def start(self):
        if self.running or not self.model:
            return

        # Show loading
        self.loading_label.config(text="Preparing video...")
        self.update_idletasks()

        # Background thread to prepare video
        def prepare_and_start():
            
            value = self.source_value.get()
            if self.source_type.get() == "Camera":
                value = int(value)
            elif self.source_type.get() == "HTTP":
                value = self.camera_url.get()
            else:
                value = get_local_video(self.source_value.get())
                if value is None:
                    self.loading_label.config(text="Video not available")
                    return
            self.after(0, lambda: self._start_inference(value))

        threading.Thread(target=prepare_and_start, daemon=True).start()
    
    def _start_inference(self, value):
        
        self.running = True
        self.loading_label.config(text="")
        self.worker_thread = threading.Thread(
                target=self.inference_worker,
                args=(value,),
                daemon=True
        )
        self.worker_thread.start()

    def stop(self):
        self.running = False
        self.count_console = 0

    # ------------------------
    # Inference Worker
    # ------------------------
    def inference_worker(self, value):
        with dgt.open_video_stream(value) as stream:
            for _, result in enumerate(dgt.predict_stream(self.model, stream)):
                if not self.running:
                    break

                self.after(
                    UI_REFRESH_MS,
                    self.update_ui_from_result,
                    result
                )

    def redis_listener(self):
        for message in self.redis_sub.listen():
            if not self.running:
                continue
            if message['type'] != 'message':
                continue
            data = message['data'].decode()
            # Example: simple command handling
            if data == "stop":
                self.stop()
            elif data.startswith("target:"):
                self.target_class.set(data.split(":")[1])

    def update_label_with_vdl(image_overlay):
        # Convert image to JPEG bytes
        _, buffer = cv2.imencode('.jpg', image_overlay)
        jpg_bytes = buffer.tobytes()
        # Publish to Redis
        try:
            self.redis_client.publish(REDIS_PUB_CHANNEL, jpg_bytes)
        except Exception as e:
            print(f"Redis publish error: {e}")

    # ------------------------
    # UI Update (Main Thread)
    # ------------------------
    def update_ui_from_result(self, result):
        if not self.running:
            return

        if self.count_console > 5:
            self.console.delete("1.0", tk.END)
            self.console.update_idletasks()

        self.count_console += 1
        
        labels = set()
        for det in result.results:
            labels.add(det.get("label", ""))

        classes = {'bottle', 'cup', 'wine glass'}  # Use {} or set([...])



        self.console.insert(tk.END, f"Detected: {labels}\n")        
        self.console.update_idletasks()

        frame = cv2.cvtColor(result.image_overlay, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame)

        # Compute scale to fit canvas while preserving aspect ratio
        video_h, video_w = frame.shape[:2]
        scale_w = VIDEO_WIDTH / video_w
        scale_h = VIDEO_HEIGHT / video_h
        scale = min(scale_w, scale_h)

        new_w = int(video_w * scale)
        new_h = int(video_h * scale)
        img_resized = img_pil.resize((new_w, new_h))

        # Create a gray background image (canvas)
        canvas_img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), color=(128, 128, 128))
        # Paste resized frame centered
        x_offset = (VIDEO_WIDTH - new_w) // 2
        y_offset = (VIDEO_HEIGHT - new_h) // 2
        canvas_img.paste(img_resized, (x_offset, y_offset))

        # Convert to Tk image
        imgtk = ImageTk.PhotoImage(canvas_img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)
        self.console.update_idletasks()

        if classes.intersection(labels):  # Non-empty set evaluates to True
            update_label_with_vdl(result.image_overlay)



# ============================
# Run
# ============================

if __name__ == "__main__":
    app = InferenceUI()
    app.mainloop()