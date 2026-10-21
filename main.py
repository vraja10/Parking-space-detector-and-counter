
import cv2
import numpy as np

from utils import get_parking_spots_bboxes, empty_or_not


def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))


mask_path = "mask_1920_1080.png"
video_path = "resized_video.mp4"
output_video_path = "output_video.mp4"

# Load mask and video
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise FileNotFoundError(f"Could not load mask: {mask_path}")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

print(f"Video resolution: {frame_width} x {frame_height}")
print(f"Mask resolution: {mask.shape[1]} x {mask.shape[0]}")

# The mask must match the video resolution.
if mask.shape[:2] != (frame_height, frame_width):
    raise ValueError(
        "Mask and video dimensions do not match. "
        "Create a new parking-space mask for this video before running detection."
    )

connected_components = cv2.connectedComponentsWithStats(
    mask, 4, cv2.CV_32S
)
spots = get_parking_spots_bboxes(connected_components)

if not spots:
    raise ValueError("No parking spaces found in the mask.")

spots_status = [None] * len(spots)
diffs = [0.0] * len(spots)

previous_frame = None
frame_nmr = 0
step = 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_video_path, fourcc, fps, (frame_width, frame_height)
)

if not out.isOpened():
    raise RuntimeError("Could not create output video.")

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_nmr % step == 0:
        if previous_frame is None:
            candidates = range(len(spots))
        else:
            for i, (x, y, w, h) in enumerate(spots):
                crop = frame[y:y + h, x:x + w]
                prev_crop = previous_frame[y:y + h, x:x + w]

                diffs[i] = calc_diff(crop, prev_crop)

            max_diff = max(diffs) if diffs else 0

            candidates = (
                [i for i in np.argsort(diffs)
                 if diffs[i] / max_diff > 0.4]
                if max_diff > 0 else []
            )

        for i in candidates:
            x, y, w, h = spots[i]
            crop = frame[y:y + h, x:x + w]

            if crop.size:
                spots_status[i] = empty_or_not(crop)

        previous_frame = frame.copy()

    for i, (x, y, w, h) in enumerate(spots):
        status = spots_status[i]

        if status is None:
            color = (255, 255, 0)
        elif status:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    available = sum(status is True for status in spots_status)

    cv2.rectangle(frame, (30, 20), (500, 80), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Available spots: {available} / {len(spots)}",
        (45, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    out.write(frame)

    # Display a smaller preview for 4K video.
    preview = cv2.resize(frame, (1280, 720))
    cv2.imshow("frame", preview)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_nmr += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved to: {output_video_path}")
