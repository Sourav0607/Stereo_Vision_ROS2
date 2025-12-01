import cv2
import numpy as np
import yaml
import os
from ultralytics import YOLO
import sys
sys.path.append('/home/sourav/cv_project/object_tracking')
from sort import Sort

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
capR = cv2.VideoCapture(4)

capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# === OBJECT TRACKING SETUP ===
MODEL_PATH = '/home/sourav/cv_project/yolov8n.pt'
CONF_THRESHOLD = 0.5

model = YOLO(MODEL_PATH)
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
print("✓ YOLO model loaded for object tracking")

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

# === Calculate baseline and focal length for depth calculation ===
# Baseline: distance between the two cameras (from translation vector)
# NOTE: Calibration T is in millimeters, convert to meters for depth calculation
baseline = np.linalg.norm(T) / 1000.0  # Convert mm to meters
# Focal length from projection matrix P1
focal_length = P1[0, 0]  # in pixels

print(f"\nCamera Setup:")
print(f"Baseline: {baseline*1000:.2f} mm ({baseline:.4f} m)")
print(f"Focal Length: {focal_length:.2f} pixels")

# Calculate theoretical max depth
theoretical_max_depth = (focal_length * baseline) / 1.0  # 1 pixel minimum disparity
practical_max_depth = (focal_length * baseline) / 5.0    # 5 pixels for reliable detection

print(f"\nDepth Range Capabilities:")
print(f"  Theoretical max: {theoretical_max_depth:.1f} m")
print(f"  Practical max: {practical_max_depth:.1f} m (reliable)")
print(f"  Recommended range: 0.5 - {practical_max_depth:.1f} m")
print(f"Q Matrix for 3D reconstruction:\n{Q}")

print("=" * 70)
print("STEREO DEPTH MAP WITH OBJECT TRACKING & DISTANCE MEASUREMENT")
print("=" * 70)
print("Controls:")
print("  ESC - Exit")
print("  's' - Save current frame")
print("  'c' - Check model accuracy")
print("  'd' - Display detailed depth statistics")
print("  't' - Toggle object tracking ON/OFF")
print("  'i' - Toggle debug info (shows confidence)")
print("  Mouse Click - Get depth at clicked point")
print("=" * 70)

frame_count = 0
mouse_x, mouse_y = -1, -1
clicked = False
tracking_enabled = True  # Object tracking ON by default
show_debug_info = False  # Debug info OFF by default

def mouse_callback(event, x, y, flags, param):
    """Mouse callback to get depth at clicked point"""
    global mouse_x, mouse_y, clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y
        clicked = True

cv2.namedWindow("Rectified (left) + Depth Map")
cv2.setMouseCallback("Rectified (left) + Depth Map", mouse_callback)

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
    
    # === DEPTH INFORMATION EXTRACTION ===
    # Convert disparity to depth (Z = focal_length * baseline / disparity)
    # Create a mask for valid disparities
    valid_mask = disp_cropped > 0
    depth_map = np.zeros_like(disp_cropped)
    depth_map[valid_mask] = (focal_length * baseline) / disp_cropped[valid_mask]
    
    # Get depth statistics
    if np.any(valid_mask):
        valid_depths = depth_map[valid_mask]
        min_depth = np.min(valid_depths)
        max_depth = np.max(valid_depths)
        mean_depth = np.mean(valid_depths)
        median_depth = np.median(valid_depths)
    else:
        min_depth = max_depth = mean_depth = median_depth = 0
    
    # Normalize for visualization
    disp_vis = cv2.normalize(disp_cropped, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    
    # Convert grayscale depth map to BGR format for consistent display with color image
    disp_gray_bgr = cv2.cvtColor(disp_vis, cv2.COLOR_GRAY2BGR)
    
    # Calculate coverage (how much has valid depth)
    valid_pixels = np.count_nonzero(disp_cropped > 0)
    total_pixels = disp_cropped.size
    coverage = (valid_pixels / total_pixels) * 100
    
    # Add info overlay
    cv2.putText(imgL_rect_cropped, f"Coverage: {coverage:.1f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(imgL_rect_cropped, f"Avg Depth: {mean_depth:.2f}m", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # === OBJECT TRACKING AND DEPTH MEASUREMENT ===
    if tracking_enabled:
        # Run YOLO detection on the cropped rectified image
        results = model(imgL_rect_cropped, conf=CONF_THRESHOLD, verbose=False)
        detections = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.cpu().numpy())
                cls = int(box.cls.cpu().numpy())
                detections.append([x1, y1, x2, y2, conf, cls])
        
        # Update tracker
        if len(detections) > 0:
            dets = np.array([[d[0], d[1], d[2], d[3], d[4]] for d in detections])
            tracked_objects = tracker.update(dets)
        else:
            tracked_objects = np.empty((0, 5))
        
        # Draw tracked objects with depth information
        for i, track in enumerate(tracked_objects):
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Calculate center of bounding box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # IMPROVED DEPTH MEASUREMENT - Sample entire object region
            if 0 <= x1 < w_roi and 0 <= y1 < h_roi and 0 <= x2 < w_roi and 0 <= y2 < h_roi:
                # Strategy 1: Get depth from entire bounding box
                bbox_depth = depth_map[y1:y2, x1:x2]
                valid_bbox = bbox_depth[bbox_depth > 0]
                
                # Strategy 2: Get depth from center region (30% of box)
                box_w, box_h = x2 - x1, y2 - y1
                center_margin_w = int(box_w * 0.35)
                center_margin_h = int(box_h * 0.35)
                cx_min = max(x1, cx - center_margin_w)
                cx_max = min(x2, cx + center_margin_w)
                cy_min = max(y1, cy - center_margin_h)
                cy_max = min(y2, cy + center_margin_h)
                
                center_depth = depth_map[cy_min:cy_max, cx_min:cx_max]
                valid_center = center_depth[center_depth > 0]
                
                # Strategy 3: Sample multiple points across the object
                sample_points = []
                grid_size = 3  # 3x3 grid of sample points
                for gy in range(grid_size):
                    for gx in range(grid_size):
                        sx = x1 + (x2 - x1) * (gx + 0.5) / grid_size
                        sy = y1 + (y2 - y1) * (gy + 0.5) / grid_size
                        sx, sy = int(sx), int(sy)
                        if 0 <= sx < w_roi and 0 <= sy < h_roi:
                            d = depth_map[sy, sx]
                            if d > 0:
                                sample_points.append(d)
                
                # Determine best depth estimate
                object_depth = None
                confidence = "low"
                
                # Prefer center region if we have enough valid points
                if len(valid_center) > 20:  # At least 20 valid pixels in center
                    # Use 25th-75th percentile to remove outliers
                    p25, p75 = np.percentile(valid_center, [25, 75])
                    filtered = valid_center[(valid_center >= p25) & (valid_center <= p75)]
                    if len(filtered) > 5:
                        object_depth = np.median(filtered)
                        confidence = "high"
                
                # Fall back to entire bounding box
                elif len(valid_bbox) > 50:  # At least 50 valid pixels in bbox
                    # Use percentile filtering to remove outliers
                    p25, p75 = np.percentile(valid_bbox, [25, 75])
                    filtered = valid_bbox[(valid_bbox >= p25) & (valid_bbox <= p75)]
                    if len(filtered) > 10:
                        object_depth = np.median(filtered)
                        confidence = "medium"
                
                # Fall back to grid sampling
                elif len(sample_points) >= 3:
                    object_depth = np.median(sample_points)
                    confidence = "low"
                
                # Format output based on confidence
                if object_depth is not None and object_depth > 0:
                    depth_text = f"{object_depth:.2f}m"
                    
                    # Color coding based on distance and confidence
                    if object_depth < 0.5:
                        depth_color = (0, 0, 255)  # Red for very close
                        depth_text = f"CLOSE! {object_depth:.2f}m"
                    elif confidence == "low":
                        depth_color = (0, 165, 255)  # Orange for low confidence
                        depth_text = f"{object_depth:.2f}m?"
                    elif confidence == "medium":
                        depth_color = (0, 255, 255)  # Yellow for medium confidence
                        depth_text = f"{object_depth:.2f}m"
                    else:  # high confidence
                        depth_color = (0, 255, 0)  # Green for good measurement
                        depth_text = f"{object_depth:.2f}m"
                    
                    # Add debug info for troubleshooting
                    debug_info = f"[{confidence[0].upper()}:{len(valid_bbox)}px]"
                else:
                    depth_color = (0, 165, 255)  # Orange
                    depth_text = "No depth"
                    debug_info = ""
            else:
                depth_color = (0, 165, 255)
                depth_text = "Out of range"
                debug_info = ""
            
            # Find the class name for this detection
            label = "Object"
            for det in detections:
                dx1, dy1, dx2, dy2, conf, cls = det
                if abs(dx1 - x1) < 10 and abs(dy1 - y1) < 10:  # Match detection to track
                    label = model.names[cls]
                    break
            
            # Draw bounding box
            cv2.rectangle(imgL_rect_cropped, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw filled background for text
            text_label = f"ID:{int(track_id)} {label}"
            (text_w, text_h), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(imgL_rect_cropped, (x1, y1 - text_h - 10), (x1 + text_w, y1), (255, 0, 0), -1)
            
            # Draw ID and class label
            cv2.putText(imgL_rect_cropped, text_label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw depth measurement below box
            cv2.putText(imgL_rect_cropped, depth_text, (x1, y2 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, depth_color, 2)
            
            # Draw debug info if enabled
            if show_debug_info and debug_info:
                cv2.putText(imgL_rect_cropped, debug_info, (x1, y2 + 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw center point
            cv2.circle(imgL_rect_cropped, (cx, cy), 4, depth_color, -1)
        
        # Display tracking status
        status_text = f"Tracking: ON ({len(tracked_objects)} objects)"
        cv2.putText(imgL_rect_cropped, status_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        # Display tracking disabled status
        cv2.putText(imgL_rect_cropped, "Tracking: OFF", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
    
    # Handle mouse click for point depth measurement
    if clicked:
        clicked = False
        # Determine if click is on left or right side
        if mouse_x < w_roi:  # Clicked on left image
            px, py = mouse_x, mouse_y
            if 0 <= px < w_roi and 0 <= py < h_roi:
                disp_val = disp_cropped[py, px]
                if disp_val > 0:
                    depth_val = depth_map[py, px]
                    # Calculate 3D point using Q matrix
                    # Adjust coordinates for ROI offset
                    point_3d = cv2.reprojectImageTo3D(np.array([[[px + x_start, py + y_start, disp_val]]], dtype=np.float32), Q)[0, 0]
                    
                    print(f"\n{'='*50}")
                    print(f"Point Information at ({px}, {py}):")
                    print(f"  Disparity: {disp_val:.2f} pixels")
                    print(f"  Depth (Z): {depth_val:.3f} m ({depth_val*100:.1f} cm)")
                    print(f"  3D Position: X={point_3d[0]:.3f}m, Y={point_3d[1]:.3f}m, Z={point_3d[2]:.3f}m")
                    print(f"{'='*50}")
                    
                    # Draw crosshair on image
                    cv2.drawMarker(imgL_rect_cropped, (px, py), (0, 255, 255), 
                                   cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(imgL_rect_cropped, f"{depth_val:.2f}m", (px+10, py-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                else:
                    print(f"No valid depth at point ({px}, {py})")
        
        mouse_x, mouse_y = -1, -1
    
    # Combine views (now properly aligned!)
    view = np.hstack((imgL_rect_cropped, disp_gray_bgr))
    cv2.imshow("Rectified (left) + Depth Map", view)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        break
    elif key == ord('s'):
        cv2.imwrite(f"depth_result_{frame_count:04d}.png", view)
        # Also save the actual depth map as numpy array
        np.save(f"depth_map_{frame_count:04d}.npy", depth_map)
        print(f"✓ Saved frame {frame_count} (image + depth data)")
        frame_count += 1
    elif key == ord('c'):
        print(f"\n{'='*50}")
        print("Model Accuracy Check:")
        print(f"  Depth coverage: {coverage:.1f}%")
        print(f"  Valid pixels: {valid_pixels}/{total_pixels}")
        print(f"  ROI size: {w_roi}x{h_roi}")
        if coverage < 30:
            print("  ⚠ Low coverage - check lighting/texture/WB")
        elif coverage < 60:
            print("  ⚠ Medium coverage - can improve")
        else:
            print("  ✓ Good coverage!")
        print(f"{'='*50}")
    elif key == ord('d'):
        print(f"\n{'='*60}")
        print("DETAILED DEPTH STATISTICS:")
        print(f"{'='*60}")
        print(f"Depth Range:")
        print(f"  Minimum depth: {min_depth:.3f} m ({min_depth*100:.1f} cm)")
        print(f"  Maximum depth: {max_depth:.3f} m ({max_depth*100:.1f} cm)")
        print(f"  Mean depth: {mean_depth:.3f} m ({mean_depth*100:.1f} cm)")
        print(f"  Median depth: {median_depth:.3f} m ({median_depth*100:.1f} cm)")
        print(f"  Depth range: {(max_depth-min_depth):.3f} m")
        print(f"\nDisparity Range:")
        print(f"  Min disparity: {np.min(disp_cropped[valid_mask]):.2f} pixels")
        print(f"  Max disparity: {np.max(disp_cropped[valid_mask]):.2f} pixels")
        print(f"\nScene Analysis:")
        # Depth zones
        near_zone = np.sum((depth_map > 0) & (depth_map < 1.0))
        mid_zone = np.sum((depth_map >= 1.0) & (depth_map < 3.0))
        far_zone = np.sum(depth_map >= 3.0)
        print(f"  Near zone (< 1m): {near_zone} pixels ({near_zone/total_pixels*100:.1f}%)")
        print(f"  Mid zone (1-3m): {mid_zone} pixels ({mid_zone/total_pixels*100:.1f}%)")
        print(f"  Far zone (> 3m): {far_zone} pixels ({far_zone/total_pixels*100:.1f}%)")
        print(f"\nCamera Parameters:")
        print(f"  Baseline: {baseline*1000:.2f} mm")
        print(f"  Focal length: {focal_length:.2f} pixels")
        print(f"  Depth resolution at 1m: {(baseline*1000)/(focal_length*1):.2f} mm")
        print(f"  Depth resolution at 2m: {(baseline*1000)/(focal_length*0.5):.2f} mm")
        print(f"{'='*60}\n")
    elif key == ord('t'):
        tracking_enabled = not tracking_enabled
        status = "ENABLED" if tracking_enabled else "DISABLED"
        print(f"\n Object Tracking {status}")
    elif key == ord('i'):
        show_debug_info = not show_debug_info
        status = "SHOWN" if show_debug_info else "HIDDEN"
        print(f"\n Debug Info {status}")

capL.release()
capR.release()
cv2.destroyAllWindows()