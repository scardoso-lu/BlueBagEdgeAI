#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk
import threading
import sys
import os
from common.toolbox import (
    resolve_net_arg,
    resolve_input_arg,
    resolve_output_resolution_arg
)
os.environ["GRAPHIC_UI"] = "1"
from object_detection.object_detection import run_inference_pipeline as run_od_pipeline
from instance_segmentation.instance_segmentation import run_inference_pipeline as run_is_pipeline

OBJECT_DETECTION_NETS = [
    "yolov10b","yolov10n","yolov10s","yolov10x",
    "yolov11l","yolov11m","yolov11n","yolov11s","yolov11x",
    "yolov5m","yolov5s","yolov6n","yolov7","yolov7x",
    "yolov8l","yolov8m","yolov8n","yolov8s","yolov8x",
    "yolov9c"
]

INSTANCE_SEG_NETS = [
    "yolov5l_seg","yolov5m_seg","yolov5n_seg","yolov5s_seg",
    "yolov8m_seg","yolov8n_seg","yolov8s_seg",
    "yolov5m_seg_with_nms","fast_sam_s"
]

class InferenceUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Unified Inference UI")
        self.geometry("800x600")
        self.stop_event = threading.Event()
        self.worker_thread = None

        # Variables
        self.pipeline_var = tk.StringVar(value="object_detection")
        self.net_var = tk.StringVar()
        self.input_var = tk.StringVar(value="camera")
        self.url_var = tk.StringVar()
        self.show_fps = tk.BooleanVar(value=False)
        self.draw_labels = tk.BooleanVar(value=True)
        self.save_output = tk.BooleanVar(value=False)

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<ButtonPress-1>", self._on_touch_start)
        canvas.bind("<B1-Motion>", self._on_touch_move)

        font = ("Arial", 18)

        def label(text):
            tk.Label(frame, text=text, font=("Arial", 20, "bold")).pack(fill="x", pady=10)

        def checkbox(text, var):
            tk.Checkbutton(frame, text=text, variable=var,
                           onvalue=True, offvalue=False,
                           font=font, padx=20, pady=10, anchor="w").pack(fill="x")

        # Pipeline
        label("Pipeline")
        pipeline_cb = ttk.Combobox(frame, textvariable=self.pipeline_var,
                                   values=["object_detection", "instance_segmentation"],
                                   state="readonly", font=font)
        pipeline_cb.pack(fill="x", padx=10)
        self.pipeline_var.trace_add("write", self._update_nets)

        # Network
        label("Network")
        self.net_cb = ttk.Combobox(frame, textvariable=self.net_var, state="readonly", font=font)
        self.net_cb.pack(fill="x", padx=10)

        # Input
        label("Input")
        ttk.Combobox(frame, textvariable=self.input_var,
                     values=["cars", "camera", "url"],
                     state="readonly", font=font).pack(fill="x", padx=10)
        tk.Entry(frame, textvariable=self.url_var, font=font).pack(fill="x", padx=10, pady=5)

        # Options
        label("Options")
        checkbox("Show FPS", self.show_fps)
        checkbox("Draw Labels", self.draw_labels)
        checkbox("Save Output", self.save_output)

        # Buttons
        tk.Button(frame, text="START", bg="green", fg="white",
                  font=("Arial", 24, "bold"), height=2,
                  command=self.start_inference).pack(fill="x", padx=20, pady=15)
        tk.Button(frame, text="RESET", bg="orange", fg="black",
                  font=("Arial", 24, "bold"), height=2,
                  command=self.reset_inference).pack(fill="x", padx=20, pady=10)

        self._update_nets()

    # ---------------- Touch scroll ----------------
    def _on_touch_start(self, event):
        event.widget.scan_mark(event.x, event.y)

    def _on_touch_move(self, event):
        event.widget.scan_dragto(event.x, event.y, gain=1)

    # ---------------- Logic ----------------
    def _update_nets(self, *_):
        nets = OBJECT_DETECTION_NETS if self.pipeline_var.get() == "object_detection" else INSTANCE_SEG_NETS
        self.net_cb["values"] = nets
        self.net_var.set(nets[0])

    def _build_params(self):
        input_value = self.input_var.get() if self.input_var.get() in ["cars", "camera"] else self.url_var.get()

        if input_value == "cars":
            input_value = "object_detection/inputs/full_mov_slow.mp4"
        match self.pipeline_var.get():
            case  "object_detection":
                return {
                    "net": resolve_net_arg(self.pipeline_var.get(), self.net_var.get(), "."),
                    "input": resolve_input_arg(self.pipeline_var.get(), input_value, os.path.join(".",self.pipeline_var.get(), "inputs" )),
                    "show_fps": self.show_fps.get(),
                    "draw_trail": self.draw_labels.get(),
                    "output_resolution": resolve_output_resolution_arg(["sd"]),
                    "enable_tracking": True,
                    "save_stream_output": self.save_output.get(),
                    "batch_size": 1,
                    "output_dir": "./output",
                    "labels" : "common/coco.txt"
                }
            case "instance_segmentation":
                return {
                    "net": resolve_net_arg(self.pipeline_var.get(), self.net_var.get(), "."),
                    "input": resolve_input_arg(self.pipeline_var.get(), input_value, os.path.join(".",self.pipeline_var.get(), "inputs" )),
                    "show_fps": self.show_fps.get(),
                    "output_resolution": resolve_output_resolution_arg(["sd"]),
                    "enable_tracking": True,
                    "save_stream_output": self.save_output.get(),
                    "batch_size": 1,
                    "output_dir": "./output",
                    'model_type': "fast", 
                    'labels_file': "common/coco.txt", 
                    'camera_resolution': ["sd"], 
                    'framerate': 30.0

                }

    def start_inference(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.stop_event.clear()
        params = self._build_params()
        print(self.pipeline_var.get())
        target = run_od_pipeline if self.pipeline_var.get() == "object_detection" else run_is_pipeline
        self.worker_thread = threading.Thread(target=target, kwargs=params, daemon=True)
        self.worker_thread.start()

    def reset_inference(self):
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=2)
        self.stop_event.clear()
        self.start_inference()


if __name__ == "__main__":
    app = InferenceUI()
    app.mainloop()
