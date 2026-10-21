
import cv2

input_video = "parking_1920_1080.mp4"
output_video = "resized_video.mp4"

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    raise ValueError("Could not open input video.")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (1920, 1080)
)

if not out.isOpened():
    cap.release()
    raise ValueError("Could not create output video.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (1920, 1080))
    out.write(frame)

cap.release()
out.release()

print("Created:", output_video)
