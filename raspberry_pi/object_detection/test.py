#!/usr/bin/env python3
import os
import sys
import queue
import threading
from functools import partial
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.tracker.byte_tracker import BYTETracker
from common.hailo_inference import HailoInfer
from object_detection_post_process import inference_result_handler
from common.toolbox import (
    init_input_source,
    get_labels,
    load_json_file,
    preprocess,
    visualize,
    FrameRateTracker,
    resolve_net_arg,
    resolve_input_arg,
    resolve_output_resolution_arg,
)

APP_NAME = Path(__file__).stem


def run_inference_pipeline(
    net,
    input,
    batch_size,
    labels,
    output_dir,
    save_stream_output=False,
    camera_resolution=None,
    output_resolution=None,
    enable_tracking=False,
    show_fps=False,
    framerate=None,
    draw_trail=False,
) -> None:
    labels = get_labels(labels)
    config_data = load_json_file("config.json")

    cap, images = init_input_source(input, batch_size, camera_resolution)

    tracker = None
    fps_tracker = None

    if show_fps:
        fps_tracker = FrameRateTracker()

    if enable_tracking:
        tracker_config = config_data.get("visualization_params", {}).get("tracker", {})
        tracker = BYTETracker(SimpleNamespace(**tracker_config))

    input_queue = queue.Queue()
    output_queue = queue.Queue()

    post_process_callback_fn = partial(
        inference_result_handler,
        labels=labels,
        config_data=config_data,
        tracker=tracker,
        draw_trail=draw_trail,
    )

    hailo_inference = HailoInfer(net, batch_size)
    height, width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess,
        args=(images, cap, framerate, batch_size, input_queue, width, height),
    )

    postprocess_thread = threading.Thread(
        target=visualize,
        args=(
            output_queue,
            cap,
            save_stream_output,
            output_dir,
            post_process_callback_fn,
            fps_tracker,
            output_resolution,
        ),
    )

    infer_thread = threading.Thread(
        target=infer,
        args=(hailo_inference, input_queue, output_queue),
    )

    preprocess_thread.start()
    postprocess_thread.start()
    infer_thread.start()

    if show_fps:
        fps_tracker.start()

    preprocess_thread.join()
    infer_thread.join()
    output_queue.put(None)
    postprocess_thread.join()

    if show_fps:
        logger.debug(fps_tracker.frame_rate_summary())

    logger.success("Inference was successful!")

    if save_stream_output or input.lower() != "camera":
        logger.success(f"Results have been saved in {output_dir}")


def infer(hailo_inference, input_queue, output_queue):
    while True:
        next_batch = input_queue.get()
        if not next_batch:
            break

        input_batch, preprocessed_batch = next_batch

        inference_callback_fn = partial(
            inference_callback,
            input_batch=input_batch,
            output_queue=output_queue,
        )

        hailo_inference.run(preprocessed_batch, inference_callback_fn)

    hailo_inference.close()


def inference_callback(
    completion_info,
    bindings_list: list,
    input_batch: list,
    output_queue: queue.Queue,
) -> None:
    if completion_info.exception:
        logger.error(f"Inference error: {completion_info.exception}")
        return

    for i, bindings in enumerate(bindings_list):
        if len(bindings._output_names) == 1:
            result = bindings.output().get_buffer()
        else:
            result = {
                name: np.expand_dims(
                    bindings.output(name).get_buffer(), axis=0
                )
                for name in bindings._output_names
            }

        output_queue.put((input_batch[i], result))


def main(
    net: str,
    input: str,
    batch_size: int = 1,
    labels: str | None = None,
    output_dir: str | None = None,
    save_stream_output: bool = False,
    camera_resolution: str | None = None,
    output_resolution=None,
    track: bool = False,
    show_fps: bool = False,
    framerate: float | None = None,
    draw_trail: bool = False,
) -> None:
    if labels is None:
        labels = str(Path(__file__).parent.parent / "common" / "coco.txt")

    if not os.path.exists(labels):
        raise FileNotFoundError(f"Labels file not found: {labels}")

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    net = resolve_net_arg(APP_NAME, net, ".")
    input = resolve_input_arg(APP_NAME, input)
    output_resolution = resolve_output_resolution_arg(output_resolution)

    run_inference_pipeline(
        net=net,
        input=input,
        batch_size=batch_size,
        labels=labels,
        output_dir=output_dir,
        save_stream_output=save_stream_output,
        camera_resolution=camera_resolution,
        output_resolution=output_resolution,
        enable_tracking=track,
        show_fps=show_fps,
        framerate=framerate,
        draw_trail=draw_trail,
    )


if __name__ == "__main__":
    # Example invocation
    main(
        net="yolov8n",
        input="http://192.168.1.145:8080/video",
        batch_size=1,
        track=True,
        show_fps=True,
    )
