#!/usr/bin/env python3
import os
import cv2
import time
import queue
import threading
import numpy as np
from functools import partial
from hailo_apps.python.core.common.hailo_inference import HailoInfer

stop_flag = False

# Press Enter to stop
def wait_for_stop():
    global stop_flag
    input("Press Enter to stop live inference...\n")
    stop_flag = True
threading.Thread(target=wait_for_stop, daemon=True).start()

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess_frame(frame, width, height):
    frame_resized = cv2.resize(frame, (width, height))
    return np.expand_dims(frame_resized, axis=0)

# ---------------------------
# Draw predictions
# ---------------------------
def draw_predictions(frame, outputs):
    class_confidences = {}
    if len(outputs) >= 1 and isinstance(outputs[0], dict):
        for out_name, out_vals in outputs[0].items():
            if out_vals.ndim == 2 and out_vals.shape[0] > 0:
                for class_idx, conf in enumerate(out_vals[0]):
                    label = f"{out_name} C{class_idx}: {conf*100:.2f}%"
                    cv2.putText(frame, label, (10, 30 + 25*class_idx),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    class_confidences[f"{out_name}_C{class_idx}"] = float(conf)
            elif out_vals.ndim == 3 and out_vals.shape[0] > 0 and out_vals.shape[2] >= 4:
                for box in out_vals[0]:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    elif len(outputs) >= 1 and outputs[0].size > 0:
        for class_idx, conf in enumerate(outputs[0][0]):
            label = f"C{class_idx}: {conf*100:.2f}%"
            cv2.putText(frame, label, (10, 30 + 25*class_idx),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            class_confidences[f"C{class_idx}"] = float(conf)
    return frame, class_confidences

# ---------------------------
# Inference callback
# ---------------------------
def inference_callback(completion_info, bindings_list, input_batch, output_queue):
    if completion_info.exception:
        print(f"Inference error: {completion_info.exception}")
        return
    for i, bindings in enumerate(bindings_list):
        if len(bindings._output_names) == 1:
            result = bindings.output().get_buffer()
        else:
            result = {name: bindings.output(name).get_buffer()
                      for name in bindings._output_names}
        try:
            output_queue.put_nowait((input_batch[i], result))
        except queue.Full:
            pass

# ---------------------------
# Capture thread
# ---------------------------
def capture_thread_fn(cap, capture_queue):
    frame_count = 0
    start_time = time.time()
    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            continue
        # Dynamic skipping: drop old frame if queue is full
        if not capture_queue.full():
            capture_queue.put(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            fps = frame_count / (time.time() - start_time)
            print(f"[Capture] FPS={fps:.1f}, Queue size={capture_queue.qsize()}")
            frame_count = 0
            start_time = time.time()
    capture_queue.put(None)

# ---------------------------
# Preprocess thread
# ---------------------------
def preprocess_thread_fn(capture_queue, input_queue, width, height):
    frame_count = 0
    start_time = time.time()
    while True:
        frame = capture_queue.get()
        if frame is None:
            break
        input_tensor = preprocess_frame(frame, width, height)
        # Dynamic skipping: drop frame if queue full
        if not input_queue.full():
            input_queue.put((frame, input_tensor))
        frame_count += 1
        if frame_count % 30 == 0:
            fps = frame_count / (time.time() - start_time)
            print(f"[Preprocess] FPS={fps:.1f}, Queue size={input_queue.qsize()}")
            frame_count = 0
            start_time = time.time()
    input_queue.put(None)

# ---------------------------
# Inference thread
# ---------------------------
def infer_thread_fn(hailo, input_queue, output_queue):
    frame_count = 0
    start_time = time.time()
    while True:
        item = input_queue.get()
        if item is None:
            break
        frame, input_tensor = item
        callback_fn = partial(inference_callback, input_batch=[frame], output_queue=output_queue)
        hailo.run(input_tensor, callback_fn)
        frame_count += 1
        if frame_count % 30 == 0:
            fps = frame_count / (time.time() - start_time)
            print(f"[Inference] FPS={fps:.1f}, Queue size={output_queue.qsize()}")
            frame_count = 0
            start_time = time.time()
    hailo.close()
    output_queue.put(None)

# ---------------------------
# Postprocess thread
# ---------------------------
def postprocess_thread_fn(output_queue, width, height, is_display, output_path):
    frame_queue = queue.Queue(maxsize=8)
    writer_thread = None

    if not is_display:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

        def writer_fn():
            while True:
                f = frame_queue.get()
                if f is None:
                    break
                writer.write(f)
            writer.release()
            print(f"Video saved to {output_path}")

        writer_thread = threading.Thread(target=writer_fn)
        writer_thread.start()
        print(f"Headless mode: saving video to {output_path}")

    fps_list = []
    prev_time = time.time()

    while True:
        item = output_queue.get()
        if item is None:
            break
        frame, outputs = item

        # If outputs is empty, still draw an empty frame
        annotated_frame, _ = draw_predictions(frame.copy(), outputs or [{}])

        # Compute FPS
        now = time.time()
        fps = 1 / (now - prev_time)
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)
        prev_time = now

        cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if is_display:
            cv2.imshow("Live Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Always try to put the frame into the writer queue
            try:
                frame_queue.put(annotated_frame, timeout=0.1)
            except queue.Full:
                pass

    if is_display:
        cv2.destroyAllWindows()
    else:
        frame_queue.put(None)  # signal writer thread to exit
        writer_thread.join()

# ---------------------------
# Main
# ---------------------------
def run_live_inference(hef_path, input_source, batch_size=1, output_video_path="output.mp4"):
    hailo = HailoInfer(hef_path, batch_size)
    height, width, _ = hailo.get_input_shape()

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input source: {input_source}")

    is_display = "DISPLAY" in os.environ and os.environ["DISPLAY"]

    capture_queue = queue.Queue(maxsize=4)
    input_queue = queue.Queue(maxsize=4)
    output_queue = queue.Queue(maxsize=4)

    threads = [
        threading.Thread(target=capture_thread_fn, args=(cap, capture_queue)),
        threading.Thread(target=preprocess_thread_fn, args=(capture_queue, input_queue, width, height)),
        threading.Thread(target=infer_thread_fn, args=(hailo, input_queue, output_queue)),
        threading.Thread(target=postprocess_thread_fn, args=(output_queue, width, height, is_display, output_video_path))
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    cap.release()

if __name__ == "__main__":
    hef_path = "resources/models/hailo8l/yolov8n.hef"
    input_source = "http://192.168.1.145:8080/video"
    run_live_inference(hef_path, input_source)
