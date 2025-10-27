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

mapLx, mapLy = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, img_size, cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, img_size, cv2.CV_32FC1)

# === Cameras ===
capL = cv2.VideoCapture(2)
capR = cv2.VideoCapture(0)

capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# === OPTIMIZED SGBM Parameters for Better Coverage ===
window_size = 5  # Smaller window = more details but needs good texture
min_disp = 0
num_disp = 16*10  # 160 (reduced from 192 for faster matching)

# Create left and right matchers for WLS filtering
stereo_left = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,           # Stricter left-right consistency
    uniquenessRatio=10,        # Increased for better uniqueness (was 5)
    speckleWindowSize=150,     # Larger speckle filtering (was 100)
    speckleRange=2,            # Stricter speckle range (was 32)
    preFilterCap=63,           # Add pre-filtering
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Right matcher for WLS filter
stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)

# WLS filter for post-processing
wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo_left)
wls_filter.setLambda(8000)      # Smoothness (higher = smoother)
wls_filter.setSigmaColor(1.5)   # Edge sensitivity

stereo = stereo_left  # Keep reference for num_disp

# === Extract valid ROI for stereo matching ===
# roi1 and roi2 from stereoRectify define the valid regions
# We'll use a common valid region that works for both
x1_L, y1_L, w1_L, h1_L = roi1
x1_R, y1_R, w1_R, h1_R = roi2

# Find common valid region (intersection)
x_start = max(x1_L, x1_R, num_disp)  # Account for disparity search range
y_start = max(y1_L, y1_R)
x_end = min(x1_L + w1_L, x1_R + w1_R, img_size[0])
y_end = min(y1_L + h1_L, y1_R + h1_R, img_size[1])

valid_roi = (x_start, y_start, x_end - x_start, y_end - y_start)
print(f"Valid stereo ROI: x={x_start}, y={y_start}, w={x_end - x_start}, h={y_end - y_start}")
print(f"Left ROI from rectify: {roi1}")
print(f"Right ROI from rectify: {roi2}")
print(f"Disparity search range: {num_disp} pixels")

print("=" * 70)
print("IMPROVED STEREO DEPTH MAP")
print("=" * 70)
print("Controls:")
print("  ESC - Exit")
print("  's' - Save current frame")
print("  'c' - Check model accuracy")
print("=" * 70)

frame_count = 0

while True:
    retL, imgL = capL.read()
    retR, imgR = capR.read()
    
    #print(imgL.shape)
    #print(imgR.shape)

    imgL = cv2.flip(imgL, 1)
    imgR = cv2.flip(imgR, 1)
    
    if not (retL and retR):
        print(" Frame grab failed")
        break
    
    # Rectify
    imgL_rect = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    imgR_rect = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)
    
    # Convert to grayscale
    grayL = cv2.cvtColor(imgL_rect, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR_rect, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better contrast
    grayL = cv2.equalizeHist(grayL)
    grayR = cv2.equalizeHist(grayR)
    
    # Compute disparity with left and right matchers
    disp_left = stereo_left.compute(grayL, grayR).astype(np.float32) / 16.0
    disp_right = stereo_right.compute(grayR, grayL).astype(np.float32) / 16.0
    
    # Apply WLS filter for better edges and occlusion handling
    disp = wls_filter.filter(disp_left, grayL, disparity_map_right=disp_right)
    
    # Mask invalid disparities
    disp[np.isnan(disp)] = 0
    disp[disp <= 1.0] = 0
    disp[disp > num_disp] = 0
    
    # Crop to valid ROI to align rectified image with depth map
    x_start, y_start, w_roi, h_roi = valid_roi
    imgL_rect_cropped = imgL_rect[y_start:y_start+h_roi, x_start:x_start+w_roi]
    disp_cropped = disp[y_start:y_start+h_roi, x_start:x_start+w_roi]
    
    # Normalize for visualization
    disp_vis = cv2.normalize(disp_cropped, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    
    # Calculate coverage (how much has valid depth)
    valid_pixels = np.count_nonzero(disp_cropped > 0)
    total_pixels = disp_cropped.size
    coverage = (valid_pixels / total_pixels) * 100
    
    # Add info overlay
    cv2.putText(imgL_rect_cropped, f"Coverage: {coverage:.1f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Combine views (now properly aligned!)
    view = np.hstack((imgL_rect_cropped, disp_color))
    cv2.imshow("Rectified (left) + Depth Map", view)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        break
    elif key == ord('s'):
        cv2.imwrite(f"depth_result_{frame_count:04d}.png", view)
        print(f" Saved frame {frame_count}")
        frame_count += 1
    elif key == ord('c'):
        print(f"\nDepth coverage: {coverage:.1f}%")
        print(f"Valid pixels: {valid_pixels}/{total_pixels}")
        print(f"ROI size: {w_roi}x{h_roi}")
        if coverage < 30:
            print(" Low coverage - check lighting/texture/WB")
        elif coverage < 60:
            print("Medium coverage - can improve")
        else:
            print(" Good coverage!")

capL.release()
capR.release()
cv2.destroyAllWindows()