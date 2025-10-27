#!/usr/bin/env python3
"""
ROS2 Stereo Point Cloud Publisher Node

This node performs complete stereo vision pipeline:
1. Captures frames from left and right USB cameras
2. Rectifies images to align epipolar lines
3. Computes disparity maps using SGBM with WLS filtering
4. Reprojects disparity to 3D point clouds with RGB colors
5. Publishes PointCloud2 messages for visualization in RViz

Topics Published:
- /stereo/points (sensor_msgs/PointCloud2): Colored 3D point cloud at ~30 FPS
- /camera/left/image_rect (sensor_msgs/Image): Rectified left camera image
- /stereo/disparity (sensor_msgs/Image): Disparity map (mono16 encoding)

Usage:
    Published quick link for visualization in RViz: ros2 run tf2_ros static_transform_publisher 0 0 -1.0  0 0 0  base_link camera_link
    ros2 run tf2_ros static_transform_publisher 0 0 0  -1.570796 0 -1.570796  camera_link left_camera_optical
    ros2 run cam_ros_node stereo_pointcloud_node

Visualization in RViz:
    Flip the point cloud along the Z-axis
    Choose  fixed frme as base_link
    Add PointCloud2 display subscribing to /stereo/points


Author: Sourav Anil Hawaldar
Date: 2025
"""

import os, cv2, yaml, numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2 as pc2

# === Optional Dependencies ===
# Open3D is optional for potential local visualization (not used in this node)
try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False

# Check if ximgproc module is available for WLS (Weighted Least Squares) filtering
# WLS significantly improves disparity map quality through edge-aware filtering
HAS_XIMGPROC = hasattr(cv2, "ximgproc")

def pack_rgb_to_float(r, g, b):

    # Shift red to bits 16-23, green to bits 8-15, blue to bits 0-7
    rgb_uint32 = (int(r) & 255) << 16 | (int(g) & 255) << 8 | (int(b) & 255)
    # Reinterpret uint32 as float32 without value conversion (bit pattern stays same)
    return np.frombuffer(np.uint32(rgb_uint32).tobytes(), dtype=np.float32)[0]

class StereoPointCloudNode(Node):
    """
    ROS2 Node for publishing stereo-derived 3D point clouds
    
    This class performs the complete stereo vision pipeline:
    1. Camera capture and rectification
    2. Stereo matching (SGBM) with optional WLS filtering
    3. 3D reprojection using Q matrix
    4. Point cloud generation with RGB colors
    5. Publishing PointCloud2 messages for RViz visualization
    
    The node publishes at ~30 FPS and handles unit conversion automatically.
    """
    def __init__(self):
        # Initialize ROS2 node with descriptive name
        super().__init__("stereo_pointcloud_node")
        
        # CvBridge converts between OpenCV images and ROS Image messages
        self.bridge = CvBridge()

        # === QoS Profile Configuration ===
        # QoS (Quality of Service) settings for reliable communication with RViz
        # RELIABLE: guarantees message delivery (vs BEST_EFFORT for lossy networks)
        # KEEP_LAST with depth=10: keeps last 10 messages in queue
        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)

        # === Create Publishers ===
        # Point cloud publisher: main output for 3D visualization
        self.pc_pub   = self.create_publisher(PointCloud2, "/stereo/points", qos)
        # Rectified image publisher: useful for debugging and visualization
        self.left_pub = self.create_publisher(Image, "/camera/left/image_rect", qos)
        # Disparity map publisher: depth visualization (closer=brighter in mono16)
        self.disp_pub = self.create_publisher(Image, "/stereo/disparity", qos)

        # === Declare and Retrieve ROS2 Parameters ===
        # Parameters allow runtime configuration without code changes
        # Usage: ros2 run pkg node --ros-args -p left_index:=2
        self.declare_parameter("left_index", 0)    # USB camera index for left camera
        self.declare_parameter("right_index", 2)   # USB camera index for right camera
        self.declare_parameter("width", 640)       # Camera resolution width (must match calibration)
        self.declare_parameter("height", 480)      # Camera resolution height (must match calibration)
        self.declare_parameter("frame_id", "left_camera_optical")  # TF coordinate frame

        # Retrieve parameter values
        self.left_index = int(self.get_parameter("left_index").value)
        self.right_index = int(self.get_parameter("right_index").value)
        self.W = int(self.get_parameter("width").value)
        self.H = int(self.get_parameter("height").value)
        self.frame_id = self.get_parameter("frame_id").value  # Used in point cloud header

        # === Load Stereo Calibration Data ===
        # Calibration files contain intrinsic (K, D) and extrinsic (R, T) parameters
        # These were generated during the stereo calibration process
        with open(os.path.expanduser('~/stereo_calib_results/left.yaml')) as f:
            left = yaml.safe_load(f)
        with open(os.path.expanduser('~/stereo_calib_results/right.yaml')) as f:
            right = yaml.safe_load(f)
        with open(os.path.expanduser('~/stereo_calib_results/stereo.yaml')) as f:
            stereo_data = yaml.safe_load(f)

        # Extract calibration parameters as numpy arrays
        self.KL, self.DL = np.array(left['K']), np.array(left['D'])    # Left: intrinsic matrix K, distortion D
        self.KR, self.DR = np.array(right['K']), np.array(right['D'])  # Right: intrinsic matrix K, distortion D
        self.R, self.T   = np.array(stereo_data['R']), np.array(stereo_data['T'])  # R: rotation, T: translation (baseline)
        self.img_size = (self.W, self.H)  # Image dimensions (width, height)

        # === Stereo Rectification ===
        # Transforms images so epipolar lines are horizontal (simplifies stereo matching)
        # Returns: R1, R2 (rectification rotations), P1, P2 (projection matrices)
        #          Q (disparity-to-depth reprojection matrix), roi1, roi2 (valid pixel regions)
        # alpha=0: crops to show only valid pixels (no black borders)
        self.R1, self.R2, self.P1, self.P2, self.Q, roi1, roi2 = cv2.stereoRectify(
            self.KL, self.DL, self.KR, self.DR, self.img_size, self.R, self.T, alpha=0
        )
        
        # Create rectification maps for efficient per-frame remapping
        # These precomputed maps are applied with cv2.remap for fast undistortion + rectification
        self.mapLx, self.mapLy = cv2.initUndistortRectifyMap(self.KL, self.DL, self.R1, self.P1, self.img_size, cv2.CV_32FC1)
        self.mapRx, self.mapRy = cv2.initUndistortRectifyMap(self.KR, self.DR, self.R2, self.P2, self.img_size, cv2.CV_32FC1)

        # === Calculate Common Valid ROI ===
        # After rectification, some image borders may be invalid (black)
        # Find intersection of valid regions from both cameras for consistent processing
        xL, yL, wL, hL = roi1  # Left camera valid region
        xR, yR, wR, hR = roi2  # Right camera valid region
        xs = max(xL, xR); ys = max(yL, yR)  # Common region starts at rightmost-left, bottommost-top
        xe = min(xL + wL, xR + wR, self.img_size[0])  # Common region ends at leftmost-right edge
        ye = min(yL + hL, yR + hR, self.img_size[1])  # Common region ends at topmost-bottom edge
        self.valid_roi = (xs, ys, max(0, xe - xs), max(0, ye - ys))  # Store as (x, y, width, height)

        # === Baseline Unit Detection and Conversion ===
        # Q matrix and 3D coordinates depend on baseline units (mm or m)
        # Auto-detect units: if baseline > 10, assume millimeters; else meters
        baseline_raw = float(np.linalg.norm(self.T))  # Euclidean distance between cameras
        self.baseline_m = baseline_raw / 1000.0 if baseline_raw > 10.0 else baseline_raw
        if baseline_raw > 10.0:
            self.get_logger().info("Detected T in millimeters; converting to meters for publishing.")

        # === Initialize USB Cameras ===
        # Open video capture devices using camera indices from parameters
        self.capL = cv2.VideoCapture(self.left_index)
        self.capR = cv2.VideoCapture(self.right_index)
        # Set resolution to match calibration (CRITICAL: must match calibration resolution)
        for cap in (self.capL, self.capR):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)

        # Verify both cameras opened successfully
        if not (self.capL.isOpened() and self.capR.isOpened()):
            self.get_logger().error("Failed to open one or both cameras.")
            raise RuntimeError("Camera open failed")

        # === Configure SGBM Stereo Matcher ===
        # SGBM (Semi-Global Block Matching) computes disparity by finding pixel correspondences
        min_disp = 0          # Minimum disparity (pixels) - typically 0
        num_disp = 16 * 10    # Number of disparities - MUST be divisible by 16 (160 total)
        block_size = 7        # Matching block size - MUST be odd, typically 5-11
        
        self.stereo_left = cv2.StereoSGBM_create(
            minDisparity=min_disp,                    # Minimum possible disparity value
            numDisparities=num_disp,                  # Maximum disparity range to search
            blockSize=block_size,                     # Size of matching window (larger = smoother, less detail)
            P1=8 * 3 * block_size ** 2,              # Penalty for small disparity changes (smoothness)
            P2=32 * 3 * block_size ** 2,             # Penalty for large disparity changes (edge preservation)
            disp12MaxDiff=1,                          # Max allowed difference in left-right consistency check
            uniquenessRatio=8,                        # Margin by which best match must beat second-best (higher = stricter)
            speckleWindowSize=80,                     # Max size of smooth disparity regions for speckle filtering
            speckleRange=2,                           # Max disparity variation within speckle window
            preFilterCap=31,                          # Truncation value for prefiltered image pixels
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY      # Use 3-way path optimization (best quality)
        )
        
        # === Setup WLS Filter (if available) ===
        # WLS (Weighted Least Squares) post-processing improves disparity quality
        # Performs left-right consistency check and edge-aware filtering
        self.use_wls = HAS_XIMGPROC
        if self.use_wls:
            # Create right matcher for consistency check (computes disparity from right camera's view)
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo_left)
            # Create WLS filter for edge-preserving smoothing
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo_left)
            self.wls_filter.setLambda(8000.0)      # Smoothness parameter (higher = smoother)
            self.wls_filter.setSigmaColor(1.5)     # Color sensitivity (lower = more edge-preserving)
            self.get_logger().info("ximgproc found → WLS enabled")
        else:
            self.get_logger().info("ximgproc not found → WLS disabled")

        # === Setup CLAHE for Contrast Enhancement ===
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) improves matching
        # Enhances local contrast in varying lighting conditions
        # clipLimit: threshold for contrast limiting (prevents noise amplification)
        # tileGridSize: size of grid for local histogram equalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        # === Create Timer for Periodic Processing ===
        # Timer at ~30 FPS: 1/30 = 0.0333 seconds per frame
        # Calls tick() method every 33ms to capture, process, and publish
        self.timer = self.create_timer(1.0/30.0, self.tick)

    def tick(self):
        """
        Timer callback to capture frames, compute point cloud, and publish.
        """
        # === Capture Frames from Cameras ===
        okL, frameL = self.capL.read()  # Left camera frame
        okR, frameR = self.capR.read()  # Right camera frame
        # Check if both captures succeeded
        if not (okL and okR):
            self.get_logger().warn("Frame grab failed"); return

        # === Rectify Images ===
        # Apply precomputed rectification maps to undistort and align images
        # This makes epipolar lines horizontal for efficient stereo matching
        left_rect  = cv2.remap(frameL, self.mapLx, self.mapLy, cv2.INTER_LINEAR)
        right_rect = cv2.remap(frameR, self.mapRx, self.mapRy, cv2.INTER_LINEAR)

        # === Preprocessing: Grayscale + CLAHE ===
        # SGBM operates on grayscale images only
        # CLAHE enhances local contrast for better matching in varied lighting
        gL = self.clahe.apply(cv2.cvtColor(left_rect,  cv2.COLOR_BGR2GRAY))
        gR = self.clahe.apply(cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY))

        # === Compute Disparity Map ===
        # SGBM finds horizontal pixel offsets between corresponding points
        # Output is 16-bit fixed-point, so divide by 16 to get actual disparity in pixels
        dispL = self.stereo_left.compute(gL, gR).astype(np.float32) / 16.0
        
        # === Apply WLS Filtering (if enabled) ===
        # WLS performs left-right consistency check and edge-aware smoothing
        if self.use_wls:
            # Compute disparity from right camera's perspective for consistency check
            dispR = self.right_matcher.compute(gR, gL).astype(np.float32) / 16.0
            # Filter using both left and right disparity maps
            disp = self.wls_filter.filter(dispL, gL, disparity_map_right=dispR)
        else:
            # Use raw disparity without filtering
            disp = dispL

        # === Clean Invalid Disparity Values ===
        # Remove NaN, Inf, and negative disparity values (invalid depth)
        disp[~np.isfinite(disp)] = 0  # Remove NaN and Inf
        disp[disp < 0.0] = 0.0         # Remove negative disparities

        # === Crop to Valid ROI ===
        # Extract only the region where both cameras have valid rectified data
        xs, ys, w_roi, h_roi = self.valid_roi
        left_crop = left_rect[ys:ys+h_roi, xs:xs+w_roi]  # Cropped rectified left image
        disp_crop = disp[ys:ys+h_roi, xs:xs+w_roi]        # Cropped disparity map

        # === Reproject Disparity to 3D Coordinates ===
        # Use Q matrix to convert 2D disparity map to 3D point cloud
        # Formula: [X Y Z W]^T = Q * [x y disparity 1]^T, then divide by W
        # Units depend on baseline units in calibration (often mm or m)
        points3d_full = cv2.reprojectImageTo3D(disp, self.Q)
        points3d_crop = points3d_full[ys:ys+h_roi, xs:xs+w_roi].copy()

        # === Create Validity Mask ===
        # Filter out invalid 3D points (no disparity or non-finite depth)
        Z = points3d_crop[:, :, 2]  # Extract Z coordinates (depth)
        valid = (disp_crop > 0) & np.isfinite(Z)  # Valid = has disparity AND finite depth

        # Early return if no valid points
        if not np.any(valid):
            return

        # === Extract RGB Colors ===
        # Get colors from rectified left image (BGR → RGB for ROS standard)
        colors_rgb = cv2.cvtColor(left_crop, cv2.COLOR_BGR2RGB)

        # === Extract Valid Points and Colors ===
        # Use mask to get only valid 3D points and their corresponding colors
        pts = points3d_crop[valid]            # shape (N,3): X, Y, Z coordinates
        cols = colors_rgb[valid]              # shape (N,3): R, G, B values (uint8)

        # === Unit Conversion: Millimeters to Meters ===
        # If Q matrix produced millimeters, convert to meters for ROS standard
        # Heuristic: if median |Z| > 20, assume millimeters → convert to meters
        z_med = float(np.median(np.abs(pts[:,2])))
        if z_med > 20.0:
            pts = pts / 1000.0  # mm → m conversion

        # Build PointCloud2
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id  # set proper optical frame if you have TF

        # Pack RGB as float32 like PCL
        rgb_f32 = np.array([pack_rgb_to_float(r, g, b) for r, g, b in cols], dtype=np.float32)

        cloud_fields = [
            PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_data = np.column_stack((pts.astype(np.float32), rgb_f32))
        pc_msg = pc2.create_cloud(header, cloud_fields, cloud_data)

        # Publish point cloud
        self.pc_pub.publish(pc_msg)

        # Also publish images (optional but useful)
        left_msg = self.bridge.cv2_to_imgmsg(left_crop, encoding='bgr8')
        left_msg.header = header
        self.left_pub.publish(left_msg)

        # Disparity as mono16 for viewers (scale by 16 like SGBM raw)
        disp16 = (np.clip(disp_crop, 0, 65535/16.0) * 16.0).astype(np.uint16)
        disp_msg = self.bridge.cv2_to_imgmsg(disp16, encoding='mono16')
        disp_msg.header = header
        self.disp_pub.publish(disp_msg)

    def destroy_node(self):
        """
        Cleanup function called when node is shutting down
        """
        try:
            # Release camera resources to free hardware for other applications
            self.capL.release()
            self.capR.release()
        except Exception:
            # Silently ignore errors (cameras may already be released)
            pass
        # Call parent class destroy_node to clean up ROS2 resources
        super().destroy_node()

def main():
 
    # Initialize ROS2 communications
    rclpy.init()
    
    node = StereoPointCloudNode()
    
    try:
       
        rclpy.spin(node)
    except KeyboardInterrupt:
     
        pass
    node.destroy_node()
    
    # Shutdown ROS2 communications
    rclpy.shutdown()

# Standard Python idiom: only run main() if script is executed directly
if __name__ == "__main__":
    main()
