#!/usr/bin/env python3
"""
Live Stereo Depth Map Calculation
Adapted to work with your webcam setup and calibration files
"""

import cv2
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt

# === Load stereo calibration ===
print("Loading calibration files...")
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

# Calculate baseline and focal length
baseline_mm = float(np.linalg.norm(T))
baseline_m = baseline_mm / 1000.0  # Convert to meters
focal_px = K_L[0, 0]  # Focal length in pixels

print(f"Baseline: {baseline_mm:.2f} mm = {baseline_m:.4f} m")
print(f"Focal length: {focal_px:.2f} pixels")
print(f"Depth formula: depth = ({focal_px:.2f} × {baseline_mm:.2f}) / disparity")

# === Rectification ===
print("\nComputing rectification maps...")
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K_L, D_L, K_R, D_R, img_size, R, T, alpha=0
)

# Create rectification maps
mapLx, mapLy = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, img_size, cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, img_size, cv2.CV_32FC1)

# === Setup cameras ===
print("\nInitializing cameras...")
CamL_id = 0  # Left camera
CamR_id = 2  # Right camera

capL = cv2.VideoCapture(CamL_id)
capR = cv2.VideoCapture(CamR_id)

capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# === Stereo matcher parameters (optimized for better coverage) ===
min_disp = 0
num_disp = 16 * 10  # 160 (must be divisible by 16)
block_size = 5      # Smaller = more detail (must be odd)

# Use WLS filtering if available
HAS_XIMGPROC = True
try:
    _ = cv2.ximgproc
except AttributeError:
    HAS_XIMGPROC = False
    print("  WLS filtering not available - install opencv-contrib-python for better results")

stereo_left = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,          # Increased from 8
    speckleWindowSize=150,       # Increased from 80
    speckleRange=2,
    preFilterCap=63,             # Increased from 31
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

if HAS_XIMGPROC:
    stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo_left)
    wls_filter.setLambda(8000.0)
    wls_filter.setSigmaColor(1.5)
    print(" WLS filtering enabled for better quality")
else:
    stereo_right = None
    wls_filter = None

def disparity_to_depth(disparity_map, baseline, focal_length):
    """
    Convert disparity map to depth map
    
    Args:
        disparity_map: Disparity values (in pixels)
        baseline: Baseline distance (in mm)
        focal_length: Focal length (in pixels)
    
    Returns:
        depth_map: Depth values (in mm)
        depth_array: Valid depth values only
    """
    # Avoid division by zero
    disparity_map[disparity_map < 1.0] = 0.001
    
    # Depth = (focal_length × baseline) / disparity
    depth_map = (focal_length * baseline) / disparity_map
    
    # Filter invalid values
    depth_map[depth_map > 10000] = 0  # Max 10 meters
    depth_map[depth_map < 100] = 0    # Min 10 cm
    
    # Get valid depth values
    depth_array = depth_map[depth_map > 0]
    
    return depth_map, depth_array

def draw_epipolar_lines(img_rect, num_lines=10):
    """Draw horizontal epipolar lines to verify rectification"""
    h, w = img_rect.shape[:2] if len(img_rect.shape) == 3 else img_rect.shape
    img_with_lines = cv2.cvtColor(img_rect, cv2.COLOR_GRAY2BGR) if len(img_rect.shape) == 2 else img_rect.copy()
    
    step = h // num_lines
    for y in range(0, h, step):
        cv2.line(img_with_lines, (0, y), (w-1, y), (0, 255, 0), 1)
    
    return img_with_lines

print("\n" + "=" * 72)
print("LIVE STEREO DEPTH MAP")
print("=" * 72)
print("Controls:")
print("  SPACE - Capture and display depth map with matplotlib")
print("  's'   - Save current disparity and depth maps")
print("  'e'   - Show epipolar lines (verify rectification)")
print("  ESC   - Exit")
print("=" * 72)

frame_count = 0
show_epipolar = False

while True:
    # Capture frames
    retL, imgL = capL.read()
    retR, imgR = capR.read()
    
    if not (retL and retR):
        print("Frame grab failed")
        break
    
    # Convert to grayscale
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    
    # Rectify images
    img1_rectified = cv2.remap(grayL, mapLx, mapLy, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(grayR, mapRx, mapRy, cv2.INTER_LINEAR)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img1_rectified = clahe.apply(img1_rectified)
    img2_rectified = clahe.apply(img2_rectified)
    
    # Compute disparity with WLS filtering
    if HAS_XIMGPROC and wls_filter is not None:
        disparity_left = stereo_left.compute(img1_rectified, img2_rectified).astype(np.float32) / 16.0
        disparity_right = stereo_right.compute(img2_rectified, img1_rectified).astype(np.float32) / 16.0
        disparity_map_unscaled = wls_filter.filter(disparity_left, img1_rectified, disparity_map_right=disparity_right)
    else:
        disparity_map_unscaled = stereo_left.compute(img1_rectified, img2_rectified).astype(np.float32) / 16.0
    
    # Filter invalid disparities
    disparity_map_unscaled[~np.isfinite(disparity_map_unscaled)] = 0
    disparity_map_unscaled[disparity_map_unscaled < 5.0] = 0
    
    # Normalize for display
    disparity_map_scaled = cv2.normalize(disparity_map_unscaled, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_scaled = np.uint8(disparity_map_scaled)
    
    # Apply colormap
    disparity_color = cv2.applyColorMap(disparity_map_scaled, cv2.COLORMAP_JET)
    
    # Show epipolar lines if enabled
    if show_epipolar:
        display_left = draw_epipolar_lines(img1_rectified)
        display_right = draw_epipolar_lines(img2_rectified)
        display = np.hstack([display_left, display_right])
        cv2.imshow("Rectified Images with Epipolar Lines", display)
    
    # Display
    coverage = 100.0 * np.count_nonzero(disparity_map_unscaled >= 5.0) / disparity_map_unscaled.size
    
    # Color code coverage
    if coverage < 30:
        color = (0, 0, 255)  # Red
        status = "LOW"
    elif coverage < 60:
        color = (0, 165, 255)  # Orange
        status = "MEDIUM"
    else:
        color = (0, 255, 0)  # Green
        status = "GOOD"
    
    cv2.putText(disparity_color, f"Coverage: {coverage:.1f}% ({status})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(disparity_color, "Press SPACE for depth map", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Tips for low coverage
    if coverage < 30:
        cv2.putText(disparity_color, "TIP: Add more light/texture", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.imshow("Disparity Map", disparity_color)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        break
    
    elif key == ord(' '):  # SPACE - Show depth map
        print("\n Computing depth map...")
        
        # Calculate depth
        depth_map, depth_array = disparity_to_depth(
            disparity_map_unscaled.copy(), 
            baseline_mm, 
            focal_px
        )
        
        # Convert to meters for display
        depth_map_m = depth_map / 1000.0
        
        # Statistics
        if len(depth_array) > 0:
            print(f"\nOVERALL DEPTH STATISTICS:")
            print(f"   Min depth: {np.min(depth_array)/1000.0:.3f} m (closest object)")
            print(f"   Max depth: {np.max(depth_array)/1000.0:.3f} m (farthest object)")
            print(f"   Mean depth: {np.mean(depth_array)/1000.0:.3f} m (average)")
            print(f"   Median depth: {np.median(depth_array)/1000.0:.3f} m  (MOST RELIABLE)")
            print(f"   Valid pixels: {len(depth_array)} / {depth_map.size} ({coverage:.1f}%)")
            
            # Center region depth (more meaningful)
            h, w = depth_map.shape
            center_y1, center_y2 = h//4, 3*h//4
            center_x1, center_x2 = w//4, 3*w//4
            center_region = depth_map[center_y1:center_y2, center_x1:center_x2]
            center_valid = center_region[(center_region > 100) & (center_region < 10000)]
            
            if len(center_valid) > 0:
                print(f"\n CENTER REGION DEPTH (main object):")
                print(f"   Median depth: {np.median(center_valid)/1000.0:.3f} m  (RECOMMENDED)")
                print(f"   Mean depth: {np.mean(center_valid)/1000.0:.3f} m")
                print(f"   Range: {np.min(center_valid)/1000.0:.3f} - {np.max(center_valid)/1000.0:.3f} m")
        
        # Create matplotlib figure
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.title('Disparity Map (Grayscale)')
        plt.imshow(disparity_map_scaled, cmap='gray')
        plt.colorbar(label='Disparity (pixels)')
        
        plt.subplot(2, 2, 2)
        plt.title('Disparity Map (Hot)')
        plt.imshow(disparity_map_scaled, cmap='hot')
        plt.colorbar(label='Disparity (pixels)')
        
        plt.subplot(2, 2, 3)
        plt.title('Depth Map (Grayscale) - Meters')
        plt.imshow(depth_map_m, cmap='gray', vmin=0, vmax=5)
        plt.colorbar(label='Depth (m)')
        
        plt.subplot(2, 2, 4)
        plt.title('Depth Map (Hot) - Meters')
        plt.imshow(depth_map_m, cmap='hot', vmin=0, vmax=5)
        plt.colorbar(label='Depth (m)')
        
        plt.tight_layout()
        plt.show()
        
        print(" Depth map displayed")
    
    elif key == ord('s'):  # Save
        cv2.imwrite(f"disparity_map_{frame_count:04d}.png", disparity_map_scaled)
        
        # Calculate and save depth
        depth_map, _ = disparity_to_depth(disparity_map_unscaled.copy(), baseline_mm, focal_px)
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_normalized = np.uint8(depth_map_normalized)
        depth_color = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
        cv2.imwrite(f"depth_map_{frame_count:04d}.png", depth_color)
        
        print(f"Saved disparity_map_{frame_count:04d}.png and depth_map_{frame_count:04d}.png")
        frame_count += 1
    
    elif key == ord('e'):  # Toggle epipolar lines
        show_epipolar = not show_epipolar
        if not show_epipolar:
            cv2.destroyWindow("Rectified Images with Epipolar Lines")
        print(f"{' Showing' if show_epipolar else ' Hiding'} epipolar lines")

# Cleanup
capL.release()
capR.release()
cv2.destroyAllWindows()
print("\n Exited successfully")
