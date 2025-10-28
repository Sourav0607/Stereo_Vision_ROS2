#!/usr/bin/env python3
"""
Stereo Vision Point Cloud Generator with Interactive Depth Measurement

This script performs real-time stereo vision processing to:
1. Capture images from left and right cameras
2. Rectify images to align epipolar lines horizontally
3. Compute disparity maps using Semi-Global Block Matching (SGBM)
4. Convert disparity to 3D point clouds with RGB colors
5. Allow interactive depth measurement via mouse clicks
6. Visualize and save colored 3D point clouds

Requirements:
- Two calibrated USB cameras
- Calibration files: left.yaml, right.yaml, stereo.yaml
- OpenCV with ximgproc (optional, for WLS filtering)
- Open3D (optional, for 3D visualization)

Author: Sourav Hawaldar
"""

import os
import cv2
import yaml
import numpy as np

# === Optional: Open3D (for point cloud view & save) ===
# Open3D is used for 3D point cloud visualization and saving PLY files
try:
    import open3d as o3d
    HAS_O3D = True
except Exception as e:
    print("[WARN] Open3D not available:", e)
    HAS_O3D = False

# === Optional: WLS (requires opencv-contrib-python) ===
# WLS (Weighted Least Squares) filter improves disparity map quality
# by performing left-right consistency check and edge-aware filtering
HAS_XIMGPROC = True
try:
    _ = cv2.ximgproc
except AttributeError:
    HAS_XIMGPROC = False

# ---------------------------
# Load stereo calibration
# ---------------------------
# Load calibration data from YAML files created during stereo calibration
# These files contain intrinsic (K, D) and extrinsic (R, T) parameters
# K = Camera intrinsic matrix (3x3): focal lengths and principal point
# D = Distortion coefficients: radial and tangential distortion
# R = Rotation matrix (3x3): relative rotation between cameras
# T = Translation vector (3x1): relative position between cameras (baseline)
with open(os.path.expanduser('~/stereo_calib_results/left.yaml')) as f:
    left = yaml.safe_load(f)
with open(os.path.expanduser('~/stereo_calib_results/right.yaml')) as f:
    right = yaml.safe_load(f)
with open(os.path.expanduser('~/stereo_calib_results/stereo.yaml')) as f:
    stereo_data = yaml.safe_load(f)

# Extract calibration parameters as numpy arrays
K_L, D_L = np.array(left['K']), np.array(left['D'])
K_R, D_R = np.array(right['K']), np.array(right['D'])
R, T = np.array(stereo_data['R']), np.array(stereo_data['T'])
img_size = (640, 480)  # Image resolution (width, height)

# Rectify (use Q from here!)
# Stereo rectification transforms images so epipolar lines are horizontal
# This makes stereo matching more efficient (search only along horizontal lines)
# Returns:
# R1, R2: Rectification transforms for left and right cameras
# P1, P2: Projection matrices in new (rectified) coordinate system
# Q: Disparity-to-depth mapping matrix (4x4) used for 3D reprojection
# roi1, roi2: Regions of Interest - valid pixels after rectification
# alpha=0: crops image to remove black borders (only valid pixels)
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K_L, D_L, K_R, D_R, img_size, R, T, alpha=0
)

# Create undistortion and rectification maps for efficient image remapping
# These maps are precomputed transformations applied to each frame
# mapLx, mapLy: Maps for left camera (x and y coordinates)
# mapRx, mapRy: Maps for right camera (x and y coordinates)
# CV_32FC1: 32-bit floating point, single channel
mapLx, mapLy = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, img_size, cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, img_size, cv2.CV_32FC1)

# ROI intersection (valid area after rectification)
# After rectification, some image borders may be invalid (black)
# Calculate the intersection of valid ROIs from both cameras
# This gives us the common area where both cameras have valid rectified pixels
xL, yL, wL, hL = roi1  # Left camera ROI: x, y, width, height
xR, yR, wR, hR = roi2  # Right camera ROI: x, y, width, height
xs = max(xL, xR)  # Common ROI start x (rightmost left edge)
ys = max(yL, yR)  # Common ROI start y (bottommost top edge)
xe = min(xL + wL, xR + wR, img_size[0])  # Common ROI end x (leftmost right edge)
ye = min(yL + hL, yR + hR, img_size[1])  # Common ROI end y (topmost bottom edge)
w_roi, h_roi = max(0, xe - xs), max(0, ye - ys)  # Common ROI dimensions
valid_roi = (xs, ys, w_roi, h_roi)  # Store as tuple for easy access

# ---------------------------
# Cameras (LEFT=0, RIGHT=4)
# ---------------------------
# Initialize video capture for both cameras
# Camera indices (0, 2) depend on your system's USB port assignment
# You may need to adjust these based on `ls /dev/video*` output
capL = cv2.VideoCapture(0)  # Left camera (typically /dev/video0)
capR = cv2.VideoCapture(2)  # Right camera (typically /dev/video2)
# Set resolution for both cameras to match calibration resolution
for cap in (capL, capR):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Width in pixels
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Height in pixels

# ---------------------------
# Stereo matcher params
# ---------------------------
# Configure Semi-Global Block Matching (SGBM) algorithm parameters
# SGBM performs stereo matching to find corresponding pixels between left/right images
min_disp = 0          # Minimum disparity (pixels) - typically 0
num_disp = 16 * 10    # Number of disparities - MUST be divisible by 16 (160 total)
block_size = 7        # Block size for matching - MUST be odd, typically 5-11

# Create SGBM stereo matcher object with optimized parameters
stereo_left = cv2.StereoSGBM_create(
    minDisparity=min_disp,                    # Minimum possible disparity value
    numDisparities=num_disp,                  # Maximum disparity range to search
    blockSize=block_size,                     # Size of matching block (larger = smoother but less detail)
    P1=8 * 3 * block_size ** 2,              # Penalty for small disparity changes (smoothness)
    P2=32 * 3 * block_size ** 2,             # Penalty for large disparity changes (preserves edges)
    disp12MaxDiff=1,                          # Max allowed difference in left-right disparity check
    uniquenessRatio=8,                        # Margin by which best match must "win" (higher = stricter)
    speckleWindowSize=80,                     # Max size of smooth disparity regions (speckle filtering)
    speckleRange=2,                           # Max disparity variation within speckle (noise removal)
    preFilterCap=31,                          # Truncation value for prefiltered image pixels
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY      # Use full-resolution 3-way SGBM algorithm
)

use_wls = HAS_XIMGPROC  # Enable WLS filtering if opencv-contrib is available
if use_wls:
    # Create right matcher for left-right consistency check
    # This computes disparity from right camera's perspective
    right_matcher = cv2.ximgproc.createRightMatcher(stereo_left)
    # Create WLS filter for post-processing disparity map
    # WLS = Weighted Least Squares - edge-aware filtering
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo_left)
    wls_filter.setLambda(8000.0)      # Smoothness parameter (higher = smoother)
    wls_filter.setSigmaColor(1.5)     # Color sensitivity (lower = more edge-preserving)
else:
    print("[INFO] ximgproc not found → WLS disabled")

# ---------------------------
# Globals for mouse picking
# ---------------------------
# Global variables for interactive depth measurement via mouse clicks
points3d_crop = None  # Stores 3D point cloud data (X, Y, Z coordinates)
disp_crop = None      # Stores disparity map (for calculating depth)
rgb_crop = None       # Stores RGB color data for point cloud
pane_w = w_roi        # Width of display pane (matches ROI width)
pane_h = h_roi        # Height of display pane (matches ROI height)
current_units = 'm'   # Current units for depth display ('m' or 'mm')

def reproject_to_3d(disp, Qmatrix):
    """
    Convert 2D disparity map to 3D coordinates using Q matrix
    
    Formula: [X Y Z W]^T = Q * [x y disparity 1]^T
    Then divide by W to get real 3D coordinates
    
    Args:
        disp: Disparity map (HxW) in pixels
        Qmatrix: 4x4 reprojection matrix from stereoRectify
    
    Returns:
        3D point cloud (HxWx3) with X, Y, Z coordinates
    """
    return cv2.reprojectImageTo3D(disp, Qmatrix)

def create_open3d_point_cloud(points, colors):
    """
    Create Open3D point cloud object from numpy arrays
    
    Args:
        points: Nx3 array of 3D coordinates (X, Y, Z)
        colors: Nx3 array of RGB colors (0.0 to 1.0 range)
    
    Returns:
        Open3D PointCloud object ready for visualization
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)   # Set point positions
    pcd.colors = o3d.utility.Vector3dVector(colors)   # Set point colors
    return pcd

def on_mouse(event, x, y, flags, param):
    """
    Mouse callback function for interactive depth measurement
    Click on the displayed image to get 3D coordinates and depth of that point
    
    Args:
        event: Mouse event type (left click, right click, etc.)
        x, y: Mouse cursor position in pixels
        flags: Additional flags (shift, ctrl, etc.)
        param: User-defined parameters (unused)
    """
    global points3d_crop, disp_crop, pane_w, pane_h, current_units
    # Only respond to left mouse button clicks
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    # Check if 3D data is available
    if points3d_crop is None or disp_crop is None:
        return

    # Determine which pane was clicked (left image or disparity map)
    # Display shows two panes side-by-side
    if x < pane_w:
        # Clicked on left pane (rectified image)
        cx, cy = x, y
    elif x < 2 * pane_w:
        # Clicked on right pane (disparity map)
        cx, cy = x - pane_w, y
    else:
        # Clicked outside valid area
        return

    # Check if click is within valid bounds
    if 0 <= cx < pane_w and 0 <= cy < pane_h:
        pt = points3d_crop[cy, cx]   # Get 3D point at clicked pixel
        d = disp_crop[cy, cx]         # Get disparity value at clicked pixel
        # Check if point has valid depth data
        if np.isfinite(pt).all() and d > 0:
            X, Y, Z = pt.tolist()     # Extract X, Y, Z coordinates
            # Handle unit conversion (Q matrix may output mm or m depending on calibration)
            if current_units == 'mm':
                X_mm, Y_mm, Z_mm = X, Y, Z
                X_m, Y_m, Z_m = X/1000.0, Y/1000.0, Z/1000.0
            else:
                X_m, Y_m, Z_m = X, Y, Z
                X_mm, Y_mm, Z_mm = X*1000.0, Y*1000.0, Z*1000.0
            
            # Depth is |Z| (distance along optical axis)
            # This is the perpendicular distance to the camera plane
            depth_m = abs(Z_m)
            # Full 3D distance from camera origin
            # This is the Euclidean distance: sqrt(X² + Y² + Z²)
            dist_m = np.linalg.norm([X_m, Y_m, Z_m])
            
            # Print measurement results to console
            print(f"\n Pixel ({cx},{cy})  disp={d:.2f}")
            print(f"   3D Position: X={X_mm:.1f} mm, Y={Y_mm:.1f} mm, Z={Z_mm:.1f} mm")
            print(f"   Depth (|Z|): {depth_m:.3f} m  |  Full distance (|P|): {dist_m:.3f} m")
        else:
            # Invalid point (no depth data or infinite value)
            print(f"\n Pixel ({cx},{cy}) invalid (no depth)")

# Create OpenCV window and register mouse callback for interactive depth measurement
cv2.namedWindow("Stereo Depth", cv2.WINDOW_NORMAL)  # Resizable window
cv2.setMouseCallback("Stereo Depth", on_mouse)      # Register mouse click handler

# Print startup information and instructions
print("=" * 72)
print("STEREO DEPTH + COLORED POINT CLOUD")
print("=" * 72)
print(f"Rectified valid ROI: x={xs}, y={ys}, w={w_roi}, h={h_roi}")
print(f"Disparity: min={min_disp}, num={num_disp}, blockSize={block_size}")
# Baseline may be in mm depending on calibration scale
# Calculate baseline distance between cameras
_baseline_raw = float(np.linalg.norm(T))  # Euclidean norm of translation vector
# Auto-detect units: if T > 10, assume millimeters; otherwise meters
baseline_m = _baseline_raw / 1000.0 if _baseline_raw > 10.0 else _baseline_raw
f_px = P1[0,0]  # Extract focal length in pixels from projection matrix
unit_note = "(T looked like mm)" if _baseline_raw > 10.0 else ""
# Display depth formula: Z = (focal_length × baseline) / disparity
print(f"Baseline ≈ {baseline_m:.4f} m {unit_note}, Focal ≈ {f_px:.1f} px  →  Z ≈ f*B / disp")
print("[Keys] SPACE: view point cloud  |  S: save PLY  |  ESC: quit")
print("=" * 72)

# Variables for point cloud saving
last_pcd = None   # Store last generated point cloud for saving
frame_id = 0      # Counter for saved point cloud filenames

# CLAHE for better local contrast
# CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances local contrast
# This improves stereo matching in scenes with varying lighting conditions
# clipLimit: Threshold for contrast limiting (prevents over-amplification of noise)
# tileGridSize: Size of grid for histogram equalization (8x8 regions)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# ========================================
# Main processing loop - runs continuously until user presses ESC
# ========================================
while True:
    # Capture frames from both cameras simultaneously
    okL, frameL = capL.read()  # Read from left camera
    okR, frameR = capR.read()  # Read from right camera
    # Check if frame capture was successful
    if not (okL and okR):
        print(" Frame grab failed")
        break

    # Rectify both images using precomputed maps
    # Rectification: removes distortion + aligns epipolar lines horizontally
    # cv2.remap applies the transformation using the precomputed mapLx, mapLy
    left_rect  = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
    right_rect = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

    # Convert to grayscale for stereo matching (SGBM works on grayscale only)
    # Apply CLAHE to improve local contrast and matching quality
    gL = cv2.cvtColor(left_rect,  cv2.COLOR_BGR2GRAY)  # BGR to grayscale
    gR = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
    gL = clahe.apply(gL)  # Adaptive histogram equalization for better contrast
    gR = clahe.apply(gR)

    # Compute disparity map using SGBM (Semi-Global Block Matching)
    # Disparity = horizontal pixel difference between corresponding points
    # Left image is reference, algorithm searches for matches in right image
    # Output is 16-bit fixed-point, so divide by 16 to get actual disparity in pixels
    dispL = stereo_left.compute(gL, gR).astype(np.float32) / 16.0

    # Apply WLS filtering for better quality (if available)
    # WLS performs left-right consistency check and edge-aware smoothing
    if use_wls:
        # Compute disparity from right camera's perspective for consistency check
        dispR = right_matcher.compute(gR, gL).astype(np.float32) / 16.0
        # Apply WLS filter using both left and right disparity maps
        disp  = wls_filter.filter(dispL, gL, disparity_map_right=dispR)
    else:
        # Use raw disparity without filtering
        disp = dispL

    # Filter invalid disparities
    # Clean up the disparity map by removing invalid values
    disp[~np.isfinite(disp)] = 0                         # Remove NaN and Inf values
    disp[disp < (min_disp + 0.0)] = 0                   # Remove disparities below minimum
    disp[disp > (min_disp + num_disp - 1)] = 0          # Remove disparities above maximum

    # Crop to common valid ROI
    # Extract only the region where both cameras have valid rectified pixels
    xs, ys, w_roi, h_roi = valid_roi
    left_crop = left_rect[ys:ys+h_roi, xs:xs+w_roi]  # Crop left image
    disp_crop = disp[ys:ys+h_roi, xs:xs+w_roi]        # Crop disparity map

    # Visualization: normalize disparity for display as image
    # Map disparity range to 0-255 for visualization
    # Step 1: Clip to valid range and normalize to [0, 1]
    disp_vis = (np.clip(disp_crop, min_disp, min_disp + num_disp) - min_disp) / float(num_disp)
    # Step 2: Scale to [0, 255] for 8-bit image
    disp_vis = (disp_vis * 255.0).astype(np.uint8)
    # Step 3: Apply JET colormap (blue=far, red=close)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # Overlay epipolar guide lines to verify rectification quality
    # If rectification is correct, corresponding points should lie on same horizontal line
    # Draw horizontal green lines every 80 pixels
    for y in range(40, h_roi, 80):
        cv2.line(left_crop, (0, y), (w_roi-1, y), (0, 255, 0), 1)

    # Calculate and display coverage metric
    # Coverage = percentage of pixels with valid disparity (depth data)
    coverage = 100.0 * np.count_nonzero(disp_crop > 0) / disp_crop.size
    cv2.putText(left_crop, f"Coverage: {coverage:.1f}%", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Create side-by-side display: left image + disparity map
    pane_w, pane_h = w_roi, h_roi  # Update pane dimensions for mouse callback
    view = np.hstack([left_crop, disp_color])  # Horizontal concatenation
    cv2.imshow("Stereo Depth", view)

    # Prepare 3D point cloud data for interactive clicking and visualization
    # Reproject entire disparity map to 3D using Q matrix
    points3d_full = reproject_to_3d(disp, Q)
    # Crop to valid ROI and create copy for thread safety
    points3d_crop = points3d_full[ys:ys+h_roi, xs:xs+w_roi].copy()
    # Convert BGR to RGB and keep in 0-255 range for Open3D
    rgb_crop = cv2.cvtColor(left_crop, cv2.COLOR_BGR2RGB)  # 0..255

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == 27:   # ESC key - exit program
        break

    elif key == ord(' '):  # SPACE key - visualize 3D point cloud
        # Check if Open3D is available
        if not HAS_O3D:
            print("[INFO] Open3D not installed → skipping point cloud view.")
            continue

        # Build mask to filter valid 3D points
        # Extract Z coordinates (depth values)
        Z = points3d_crop[:, :, 2]
        # Create mask for valid Z values: has disparity and is finite
        validZ = (disp_crop > 0) & np.isfinite(Z)
        # Auto-detect units (mm or m) by checking median Z value
        if np.any(validZ):
            z_med = float(np.median(np.abs(Z[validZ])))
            # If median Z > 20, assume millimeters; else meters
            current_units = 'mm' if z_med > 20.0 else 'm'
        else:
            current_units = 'm'

        # Filter points based on reasonable depth range for current units
        if current_units == 'mm':
            # Range: 10mm (1cm) to 10000mm (10m)
            mask = validZ & (np.abs(Z) > 10.0) & (np.abs(Z) < 10000.0)
        else:
            # Range: 0.01m (1cm) to 10m
            mask = validZ & (np.abs(Z) > 0.01) & (np.abs(Z) < 10.0)
        # Extract points and colors using mask
        pts = points3d_crop[mask]
        cols = (rgb_crop[mask].astype(np.float32) / 255.0)  # Normalize colors to [0, 1]

        # Print point cloud statistics
        if len(pts) > 0:
            Zv = Z[mask]  # Get Z values for masked points
            print(f"\n Point cloud: {len(pts)} points | units={current_units} | disp range: "
                  f"{disp_crop[disp_crop>0].min() if np.any(disp_crop>0) else 0:.2f}.."
                  f"{disp_crop.max():.2f}, coverage={coverage:.1f}%")
            print(f"   Z stats: min={np.min(Zv):.2f} {current_units}, med={np.median(Zv):.2f} {current_units}, max={np.max(Zv):.2f} {current_units}")
        else:
            print("\n Point cloud: 0 points | attempting relaxed thresholds...")

        # If too few points, relax thresholds and try again
        if len(pts) < 100:
            # Relax depth range constraints
            if current_units == 'mm':
                mask = validZ & (np.abs(Z) > 1.0) & (np.abs(Z) < 20000.0)  # 1mm to 20m
            else:
                mask = validZ & (np.abs(Z) > 0.001) & (np.abs(Z) < 20.0)  # 1mm to 20m
            pts = points3d_crop[mask]
            cols = (rgb_crop[mask].astype(np.float32) / 255.0)
            print(f"   → After relaxing: {len(pts)} points")
            # If still too few points, skip visualization
            if len(pts) < 100:
                print("   Still too few points. Ensure good texture/lighting and objects within range.")
                continue

        # Create Open3D point cloud object
        pcd = create_open3d_point_cloud(pts, cols)
        last_pcd = pcd  # Store for potential saving

        # Add coordinate frame for reference (X=red, Y=green, Z=blue)
        # Size=0.3 means each axis is 0.3 units long
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0,0,0])
        # Visualize point cloud and coordinate frame together
        o3d.visualization.draw_geometries(
            [pcd, frame],
            window_name="Colored Point Cloud",
            width=1024, height=768
        )
        print("Closed viewer")

    elif key == ord('s'):  # 's' key - save point cloud to PLY file
        # Check if Open3D is available and a point cloud has been generated
        if HAS_O3D and last_pcd is not None:
            # Generate filename with sequential numbering
            fname = f"point_cloud_{frame_id:04d}.ply"
            # Save in PLY format (Polygon File Format - stores 3D data)
            o3d.io.write_point_cloud(fname, last_pcd)
            print(f" Saved: {fname}")
            frame_id += 1  # Increment counter for next save
        else:
            print(" No point cloud to save. Press SPACE first.")

# ========================================
# Cleanup and exit
# ========================================
# Release camera resources
capL.release()
capR.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
