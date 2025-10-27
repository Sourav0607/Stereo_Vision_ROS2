import cv2
import numpy as np
import yaml
import os

# === Load calibration ===
with open(os.path.expanduser('~/stereo_calib_results/left.yaml')) as f: 
    left = yaml.safe_load(f)
with open(os.path.expanduser('~/stereo_calib_results/right.yaml')) as f: 
    right = yaml.safe_load(f)
with open(os.path.expanduser('~/stereo_calib_results/stereo.yaml')) as f: 
    stereo_data = yaml.safe_load(f)

K_L, D_L = np.array(left['K']), np.array(left['D'])
K_R, D_R = np.array(right['K']), np.array(right['D'])
R, T = np.array(stereo_data['R']), np.array(stereo_data['T'])

img_size = (640, 480)

# === Rectification ===
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K_L, D_L, K_R, D_R, img_size, R, T, alpha=0
)

mapLx, mapLy = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, img_size, cv2.CV_32FC1) # Left map
mapRx, mapRy = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, img_size, cv2.CV_32FC1) # Right map

# === Cameras ===
capL = cv2.VideoCapture(2)
capR = cv2.VideoCapture(0)

capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# === OPTIMIZED SGBM for Better Real Coverage ===
window_size = 5                                        # Smaller window = more details but needs good texture
min_disp = 0                        # Minimum disparity
num_disp = 16*12         # 160 disparities (reduced from 192 for faster matching)

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=-1,              # Disable left-right check for more matches
    uniquenessRatio=5,             # LOWER for more matches (was 10)
    speckleWindowSize=100,         # Moderate filtering (was 150)
    speckleRange=32,               # HIGHER to accept more variation (was 2)
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# ROI calculation
x1_L, y1_L, w1_L, h1_L = roi1
x1_R, y1_R, w1_R, h1_R = roi2
x_start = max(x1_L, x1_R, num_disp)
y_start = max(y1_L, y1_R)
x_end = min(x1_L + w1_L, x1_R + w1_R, img_size[0])
y_end = min(y1_L + h1_L, y1_R + h1_R, img_size[1])
valid_roi = (x_start, y_start, x_end - x_start, y_end - y_start)

print("=" * 80)
print("STEREO DEPTH MAP - RAW SGBM")
print("=" * 80)
print("Press 's' to save current frame")
print("Press ESC to exit")
print("=" * 80)

while True:
    retL, imgL = capL.read()
    retR, imgR = capR.read()
    
    imgL = cv2.flip(imgL, 1)
    imgR = cv2.flip(imgR, 1)
    
    if not (retL and retR):
        break
    
    # Rectify
    imgL_rect = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    imgR_rect = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)
    
    # Grayscale + preprocessing for better matching
    grayL = cv2.cvtColor(imgL_rect, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR_rect, cv2.COLOR_BGR2GRAY)
    
    # Histogram equalization for better contrast
    grayL = cv2.equalizeHist(grayL)
    grayR = cv2.equalizeHist(grayR)
    
    # Optional: Apply slight gaussian blur to reduce noise before matching
    grayL = cv2.GaussianBlur(grayL, (5, 5), 0)
    grayR = cv2.GaussianBlur(grayR, (5, 5), 0)
    
    # Compute disparity
    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    
    # Mask invalid disparities (but keep threshold lower for more coverage)
    disp[disp < 0] = 0
    disp[disp > num_disp] = 0
    
    # Apply bilateral filter instead of median for edge-preserving smoothing
    disp = cv2.bilateralFilter(disp.astype(np.float32), 5, 50, 50)
    
    # Crop to ROI
    x_start, y_start, w_roi, h_roi = valid_roi
    imgL_rect_cropped = imgL_rect[y_start:y_start+h_roi, x_start:x_start+w_roi]
    disp_cropped = disp[y_start:y_start+h_roi, x_start:x_start+w_roi]
    
    # Coverage calculation
    valid_pixels = np.count_nonzero(disp_cropped > 0)
    total_pixels = disp_cropped.size
    coverage = (valid_pixels / total_pixels) * 100
    
    # Visualize
    disp_vis = cv2.normalize(disp_cropped, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    
    # Add overlay
    cv2.putText(imgL_rect_cropped, "RAW SGBM", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(imgL_rect_cropped, f"Coverage: {coverage:.1f}%", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display
    view = np.hstack((imgL_rect_cropped, disp_color))
    cv2.imshow("Stereo Depth Map", view)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        break
    elif key == ord('s'):
        cv2.imwrite(f"depth_result.png", view)
        print(f"\n Saved frame!")
        print(f"   Coverage: {coverage:.1f}%")

capL.release()
capR.release()
cv2.destroyAllWindows()
