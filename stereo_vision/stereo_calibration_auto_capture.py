import cv2
import os
import time
from datetime import datetime

# Camera device IDs
left_id = 4    # /dev/video4
right_id = 0   # /dev/video0

# Capture settings
cap_left = cv2.VideoCapture(left_id)
cap_right = cv2.VideoCapture(right_id)

cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Directory setup
save_dir = os.path.expanduser("~/stereo_calib_images_auto")
os.makedirs(os.path.join(save_dir, "left"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "right"), exist_ok=True)

img_counter = 0
delay = 3.0  # seconds between captures
last_capture = time.time()

print(" Automatic stereo capture started...")
print("Every 3 seconds, one pair will be saved.")
print("Press ESC to stop.")

while True:
    retL, frameL = cap_left.read()
    retR, frameR = cap_right.read()
    if not (retL and retR):
        print(" Could not read both cameras.")
        break

    # Optional: flip horizontally if mirrored
    frameL = cv2.flip(frameL, 1)
    frameR = cv2.flip(frameR, 1)

    # Combine for live view
    both = cv2.hconcat([frameL, frameR])
    cv2.imshow("Left | Right", both)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key to stop
        print(" Stopped capturing.")
        break

    # Auto-capture every N seconds
    if time.time() - last_capture >= delay:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        left_filename = os.path.join(save_dir, "left", f"left_{img_counter:03d}.png")
        right_filename = os.path.join(save_dir, "right", f"right_{img_counter:03d}.png")

        cv2.imwrite(left_filename, frameL)
        cv2.imwrite(right_filename, frameR)
        print(f"Saved pair {img_counter} at {timestamp}")
        img_counter += 1
        last_capture = time.time()

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
