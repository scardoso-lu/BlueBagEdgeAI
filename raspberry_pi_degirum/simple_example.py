import degirum as dg
import degirum_tools as dgt
import cv2

# Declaring variables
# Set your model name, inference host address, model zoo, and AI Hub token.
#model_name="yolo11n_coco--640x640_quant_hailort_multidevice_1"
#model_name="yolov8n_relu6_coco_seg--640x640_quant_hailort_hailo8l_1"
#model_name="yolov8n_relu6_coco_pose--640x640_quant_hailort_hailo8l_1"
#model_name="yolov8n_relu6_lp_ocr--256x128_quant_hailort_multidevice_1"
model_name="yolov8n_relu6_face--640x640_quant_hailort_multidevice_1"
zoo_url="degirum/hailo"
host_address = "@local" # Can be "@cloud", host:port, or "@local"


# Loading a model
model = dg.load_model(
    model_name = model_name, 
    inference_host_address = host_address, 
    zoo_url = zoo_url, 
    # optional parameters, such as output_confidence_threshold=0.5
)
# ----------------------------
# Configuration
# ----------------------------
LOCAL_VIDEO = "videos/Traffic2.mp4"

# Optional tuning
# Enable overlays BEFORE starting the stream
model.overlay_show_labels = True
model.overlay_show_confidence = True
model.overlay_show_bbox = True   # <-- important for boxes!
model.confidence_threshold = 0.4

# ----------------------------
# Open YouTube stream
# ----------------------------
with dgt.open_video_stream(LOCAL_VIDEO) as stream:

    for frame_id, result in enumerate(dgt.predict_stream(model, stream)):


        # result.results → raw detections
        for det in result.results:
            if len(det) > 0:
                print(
                    f"Class: {det['label']}, "
                    f"BBox: {det['bbox']}"
                )

        # Display
        cv2.imshow("DeGirum Inference", result.image_overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
