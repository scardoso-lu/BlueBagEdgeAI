import cv2
import time

VIDEO_STREAM = "http://192.168.1.145:8080/video"
cap = cv2.VideoCapture(VIDEO_STREAM)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Live Inference", frame)

    # FPS calculation
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    print(f"FPS: {fps:.2f}")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done capturing frames")
