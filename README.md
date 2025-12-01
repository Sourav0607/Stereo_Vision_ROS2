# Stereo Vision ROS2

A complete stereo vision system for 3D reconstruction and depth estimation using dual USB cameras, OpenCV, and ROS2 integration.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble+-orange.svg)](https://docs.ros.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-red.svg)](https://opencv.org/)

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start Guide](#quick-start-guide)
- [Stereo Calibration](#stereo-calibration)
- [Depth Estimation](#depth-estimation)
- [3D Point Cloud Visualization](#3d-point-cloud-visualization)
- [ROS2 Integration](#ros2-integration)
- [Technical Specifications](#technical-specifications)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a complete stereo vision pipeline for real-time 3D reconstruction using two USB webcams. It includes camera calibration, stereo rectification, disparity mapping using Semi-Global Block Matching (SGBM), and 3D point cloud generation with RGB coloring.

### Key Capabilities

- **Stereo Camera Calibration**: Automated calibration using checkerboard patterns
- **Real-time Depth Estimation**: SGBM with WLS filtering for high-quality disparity maps
- **3D Point Cloud Generation**: Colored point clouds with Open3D visualization
- **ROS2 Integration**: Publishes PointCloud2 messages for RViz visualization
- **Interactive Depth Measurement**: Click-to-measure depth functionality

##  Features

### Calibration Tools
-  Automatic stereo calibration with checkerboard detection
-  Manual and auto-capture modes for calibration images
-  Reprojection error analysis (achieved: **0.57 pixels**)
-  YAML export of calibration parameters

### Depth Estimation & Object Tracking
-  **Real-time object detection and tracking** with YOLO v8 + SORT
-  **Distance measurement for tracked objects** with confidence indicators
-  **Multi-strategy depth sampling** for robust measurements
-  **75% depth map coverage** with optimized parameters
-  SGBM (Semi-Global Block Matching) stereo matching
-  WLS (Weighted Least Squares) filtering for edge-aware smoothing
-  CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing
-  Real-time disparity visualization (grayscale or colored)
-  Interactive click-to-measure depth functionality

### 3D Visualization
-  Colored point cloud generation from stereo images
-  Open3D integration for interactive 3D viewing
-  PLY file export for external processing
-  Coordinate frame visualization
-  Mouse-click depth measurement
-  Depth statistics and analysis tools

### ROS2 Integration
-  PointCloud2 publisher for RViz visualization
-  Rectified image and disparity map topics
-  Configurable camera parameters via ROS2 params
-  ~30 FPS real-time performance

## System Requirements

### Hardware
- **Cameras**: 2x USB webcams (tested with 640x480 resolution)
- **Baseline**: ~115mm between camera centers (measured: **114.79mm**)
- **RAM**: 4GB minimum, 8GB recommended
- **OS**: Ubuntu 20.04+ (tested on Ubuntu 22.04)

![Stereo Camera Setup](images/camera_setup.jpg)
*Our stereo camera rig: Two UGREEN USB webcams mounted with ~115mm baseline separation*

### Software
- **Python**: 3.8 or higher
- **ROS2**: Humble Hawksbill or later (optional, for ROS2 features)
- **OpenCV**: 4.5+ with contrib modules (ximgproc for WLS filtering)
- **Open3D**: 0.13+ for 3D visualization
- **NumPy**: 1.19+
- **Ultralytics**: YOLO v8 for object detection
- **SORT**: Simple Online Realtime Tracker for object tracking

##  Installation

### 1. Install System Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip -y

# Install OpenCV dependencies
sudo apt install libopencv-dev python3-opencv -y

# Install v4l-utils (for camera debugging)
sudo apt install v4l-utils -y
```

### 2. Install Python Packages

```bash
# Install core dependencies
pip3 install opencv-contrib-python numpy pyyaml

# Install Open3D for 3D visualization
pip3 install open3d

# Install matplotlib for plotting
pip3 install matplotlib

# Install YOLO and tracking for object detection
pip3 install ultralytics
```

### 3. Install ROS2 (Optional - for ROS2 features)

```bash
# Install ROS2 Humble (Ubuntu 22.04)
sudo apt install ros-humble-desktop -y

# Install ROS2 Python dependencies
pip3 install sensor_msgs_py

# Install cv_bridge
sudo apt install ros-humble-cv-bridge -y
```

### 4. Clone the Repository

```bash
# Clone the project
git clone https://github.com/Sourav0607/Stereo_Vision_ROS2.git
cd Stereo_Vision_ROS2

# Create calibration results directory
mkdir -p ~/stereo_calib_results
```

##  Project Structure

```
Stereo_Vision_ROS2/
│
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
├── LICENSE                            # MIT License
├── yolov8n.pt                         # YOLO model weights
│
├── stereo_vision/                     # Core stereo vision scripts
│   ├── stereo_calibrate.py           # Manual stereo calibration
│   ├── stereo_calibration_auto_capture.py  # Auto-capture calibration
│   ├── point_cloud_3d.py             # 3D point cloud visualization
│   ├── depth_map_wsl.py              # Depth map + Object tracking + Distance
│   ├── depth_trial_without_wsl.py    # Basic depth map (no WLS)
│   ├── rectification_test.py         # Rectification verification
│   ├── verify_usbport_cameraL.py     # Left camera USB detection
│   └── verify_usbport_cameraR.py     # Right camera USB detection
│
├── object_tracking/                   # Object detection & tracking
│   ├── object_tracking.py            # YOLO + SORT tracking
│   └── sort.py                       # SORT tracker implementation
│
├── images/                            # Project images and photos
│   └── camera_setup.jpg              # Stereo camera rig photo
│
├── cam_ros_node/                      # ROS2 integration package
│   └── cam_ros_node/
│       ├── setup.py                   # ROS2 package configuration
│       ├── package.xml                # ROS2 package manifest
│       └── cam_ros_node/
│           ├── cam_ros_node.py       # Camera image publisher
│           └── stereo_pointcloud_node.py  # Point cloud publisher
│
└── Visualisation_outputs/             # Sample outputs and results
    ├── point_cloud_*.ply             # Saved point cloud files
    └── calibration_images/           # Calibration image captures
```

##  Quick Start Guide

### Step 1: Verify Camera Connections

```bash
# List available video devices
ls /dev/video*

# Test left camera (adjust index if needed)
python3 stereo_vision/verify_usbport_cameraL.py

# Test right camera
python3 stereo_vision/verify_usbport_cameraR.py
```

**Expected Output**: Live video feed from each camera. Press 'q' to exit.

### Step 2: Perform Stereo Calibration

####  Auto Capture Mode (After every 3 seconds)

```bash
cd stereo_vision
python3 stereo_calibrate_auto capture.py
```

####  Calibration of Camera

```bash
python3 stereo_calibration.py
```
**Calibration Results**: Saved to `~/stereo_calib_results/`
- `left.yaml` - Left camera intrinsics (K, D)
- `right.yaml` - Right camera intrinsics (K, D)
- `stereo.yaml` - Stereo extrinsics (R, T, baseline)

### Step 3: Generate 3D Point Cloud

```bash
python3 stereo_vision/point_cloud_3d.py
```

**Controls**:
- **Click on image**: Display 3D coordinates and depth at pixel
- **SPACE**: Open Open3D viewer with colored point cloud
- **ESC**: Exit application

**Mouse Controls in Open3D**:
- **Left drag**: Rotate view
- **Right drag**: Pan view
- **Scroll**: Zoom in/out
- **R**: Reset view

### Step 4: Object Tracking with Distance Measurement

```bash
python3 stereo_vision/depth_map_wsl.py
```

**Controls**:
- **ESC**: Exit application
- **t**: Toggle object tracking ON/OFF
- **i**: Toggle debug info (shows confidence levels)
- **s**: Save current frame + depth data (.npy)
- **c**: Check model accuracy statistics
- **d**: Display detailed depth statistics
- **Click**: Get depth at any point

**Features**:
- Real-time object detection and tracking with YOLO v8
- Distance measurement for each tracked object
- Color-coded confidence indicators:
  - 🟢 **Green** = High confidence (many valid pixels)
  - 🟡 **Yellow** = Medium confidence (decent samples)
  - 🟠 **Orange with "?"** = Low confidence (few pixels)
  - 🔴 **Red** = Very close (<0.5m warning)

##  Results & Visualizations

Here are some example outputs from our stereo vision system:

### Depth Map with Rectified Image
![Rectified Image and Depth Map](Visualisation_outputs/Rectified%20(left)%20+%20Depth%20Map.png)
*Left: Rectified camera image | Right: Disparity/depth map with JET colormap (blue=far, red=close)*


### Depth Map Comparison without WLS
![Depth Map without WLS](Visualisation_outputs/Stereo%20Depth%20map_without_wsl.png)
*Stereo depth map showing disparity distribution and coverage metrics*

These results demonstrate:
- **75% depth map coverage** - High quality stereo matching
- **Smooth disparity maps** - Effective WLS filtering reduces noise
- **Colored point clouds** - RGB texture mapped onto 3D geometry
- **Real-time performance** - Processing at ~30 FPS

##  Stereo Calibration

### Checkerboard Specifications

Our calibration uses:
- **Pattern**: 8×6 internal corners (9×7 squares)
- **Square Size**: 30mm × 30mm
- **Material**: Printed on flat, rigid surface

### Calibration Quality Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Reprojection Error | 0.5737 pixels |  Excellent |
| Baseline Distance | 114.79 mm |  Measured |
| Focal Length | 1013.87 pixels | Calibrated |
| Coverage | 75% |  High |

**Reprojection Error Interpretation**:
- < 0.5 pixels: Excellent
- 0.5-1.0 pixels: Good (our result: **0.57**)
- 1.0-2.0 pixels: Acceptable
- \> 2.0 pixels: Poor, recalibrate

### Calibration Tips

1. **Lighting**: Use uniform, diffuse lighting (avoid shadows and glare)
2. **Coverage**: Capture images with checkerboard at:
   - Different depths (near and far)
   - Different angles (tilted, rotated)
   - All corners of the field of view
3. **Focus**: Ensure both cameras are in focus
4. **Stability**: Keep cameras rigidly mounted during capture
5. **Quantity**: Capture 20-30 image pairs for robust calibration

##  Depth Estimation

> ⚠️ **IMPORTANT NOTE ON DEPTH ACCURACY:**
> 
> **Calibration vs. Measurement Quality**
> 
> While our stereo system has **excellent calibration** (0.57 pixel reprojection error), the **depth measurements can still show inconsistencies** due to fundamental limitations of stereo vision with consumer-grade cameras:
>
> **Why Objects at Same Distance May Show Different Depths:**
> 
> 1. **Depth Map Quality Variations** 
>    - Stereo matching produces a depth map where some pixels have valid depth, others don't
>    - Small objects may fall partially on "depth holes" (black regions)
>    - The measured depth depends on which pixels are sampled
> 
> 2. **Low-Quality Consumer Cameras** 
>    - Rolling shutter (not global shutter) causes motion artifacts
>    - Auto-exposure/auto-white-balance changes between frames
>    - Lower resolution (640×480) limits disparity precision
>    - Noise at low light conditions
> 
> 3. **Environmental Factors** 
>    - **Textureless surfaces**: Smooth walls, bottles → Poor stereo matching
>    - **Reflective surfaces**: Glass, metal → Invalid depth data
>    - **Poor lighting**: Shadows, low contrast → Noisy depth map
>    - **Small objects**: Few pixels with valid depth → Lower confidence
> 
> 4. **Distance-Dependent Accuracy** 
>    - At **1-3m**: ±0.9-7.7 cm error (excellent)
>    - At **5m**: ±21.5 cm error (acceptable)
>    - At **7-10m**: ±36-86 cm error (marginal)
>    - Beyond **10m**: Unreliable with this camera setup
> 
> **Our Solution: Multi-Strategy Depth Sampling**
> 
> The updated code uses three sampling strategies to improve robustness:
> - ✅ Samples entire object region (not just center point)
> - ✅ Removes outliers using statistical filtering
> - ✅ Shows confidence indicators (High/Medium/Low)
> - ✅ Falls back gracefully when data is poor
> 
> **Expected Results:**
> - Objects on the same shelf should show depths within ±20-30 cm
> - Confidence indicators help you trust the measurements
> - For high-precision applications, consider upgrading to:
>   - Industrial cameras with global shutter
>   - Higher resolution (1920×1080 or better)
>   - Larger baseline (200-300mm for long range)
>   - Active stereo (structured light/time-of-flight)

### Depth Formula

Our system uses the standard stereo vision equation:

```
Z = (focal_length × baseline) / disparity
Z = (1013.87 px × 114.79 mm) / disparity
Z ≈ 116,379 / disparity (in mm)
```

**Example Calculations**:
- Disparity = 160 pixels → Depth = 0.73 m (minimum depth)
- Disparity = 50 pixels → Depth = 2.33 m (optimal)
- Disparity = 16 pixels → Depth = 7.27 m (your shelf example)
- Disparity = 5 pixels → Depth = 23.3 m (practical maximum)
- Disparity = 1 pixel → Depth = 116 m (theoretical, unreliable)

### Disparity Range

Our depth range with 114.85mm baseline and 1013.87px focal length:

| Disparity | Depth | Accuracy (±1px) | Category | Notes |
|-----------|-------|-----------------|----------|-------|
| 160 px | 0.73 m | ±0.3 cm | **Minimum** | Closer objects fall outside search range |
| 116 px | 1.0 m | ±0.9 cm | Excellent | Best accuracy zone |
| 58 px | 2.0 m | ±3.4 cm | Very Good | Optimal working range |
| 39 px | 3.0 m | ±7.7 cm | Good | Still reliable |
| 23 px | 5.0 m | ±21.5 cm | Acceptable | Accuracy decreasing |
| **16 px** | **7.3 m** | **±36 cm** | **Marginal** | **Your shelf distance** |
| 12 px | 10.0 m | ±86 cm | Poor | Near practical limit |
| 5 px | 23.3 m | ±4.3 m | **Maximum** | Practical limit for reliability |
| 1 px | 116 m | N/A | Theoretical | Unrealistic, cannot measure |

**Recommended Working Range**: 0.73m - 23.3m (full range), 1m - 5m (optimal accuracy)

### SGBM Parameters

Our optimized SGBM parameters for 75% coverage:

```python
minDisparity = 0              # Start of disparity search
numDisparities = 160          # Range of disparity search (must be ÷16)
blockSize = 7                 # Matching window size (odd number)
P1 = 8 × 3 × blockSize²      # Small disparity smoothness penalty
P2 = 32 × 3 × blockSize²     # Large disparity smoothness penalty
uniquenessRatio = 8          # Match confidence threshold
speckleWindowSize = 80       # Speckle filter window
speckleRange = 2             # Max disparity in speckle region
```

### WLS Filtering

WLS (Weighted Least Squares) post-processing improves disparity quality:

```python
lambda = 8000.0              # Smoothness (higher = smoother)
sigma_color = 1.5            # Edge sensitivity (lower = sharper edges)
```

**Benefits**:
- Removes noise and artifacts
- Preserves depth discontinuities at object edges
- Fills small holes in disparity map
- Improves overall accuracy

##  3D Point Cloud Visualization

### Point Cloud Generation Pipeline

1. **Rectify Images**: Align epipolar lines horizontally
2. **Compute Disparity**: SGBM matching + WLS filtering
3. **3D Reprojection**: Use Q matrix to convert disparity → 3D
4. **Color Mapping**: Extract RGB from rectified left image
5. **Filtering**: Remove invalid points (no disparity, non-finite depth)

### Q Matrix Reprojection

The Q matrix transforms 2D disparity to 3D coordinates:

```
[X]       [x]
[Y]   = Q [y]
[Z]       [d]
[W]       [1]
```

Then normalize: `(X/W, Y/W, Z/W)` → final 3D point

##  ROS2 Integration

### Building the ROS2 Package

```bash
# Navigate to workspace
cd ~/Stereo_Vision_ROS2

# Source ROS2
source /opt/ros/humble/setup.bash

# Build the package
colcon build --packages-select cam_ros_node

# Source the workspace
source install/setup.bash
```

### Running ROS2 Nodes

#### 1. Camera Image Publisher

```bash
ros2 run cam_ros_node cam_ros_node
```

**Published Topics**:
- `/camera/left/image_raw` - Left camera BGR images
- `/camera/right/image_raw` - Right camera BGR images

#### 2. Stereo Point Cloud Publisher

```bash
ros2 run cam_ros_node stereo_pointcloud_node
```

**Published Topics**:
- `/stereo/points` - PointCloud2 (colored 3D points)
- `/camera/left/image_rect` - Rectified left image
- `/stereo/disparity` - Disparity map (mono16)

**Parameters**:
```bash
# Change camera indices
ros2 run cam_ros_node stereo_pointcloud_node --ros-args \
  -p left_index:=2 -p right_index:=0

# Change resolution
ros2 run cam_ros_node stereo_pointcloud_node --ros-args \
  -p width:=1280 -p height:=720

# Change TF frame
ros2 run cam_ros_node stereo_pointcloud_node --ros-args \
  -p frame_id:="camera_optical_frame"
```

### Visualizing in RViz

```bash
# Launch RViz
rviz2

# In RViz:
# 1. Set Fixed Frame to: "base_link"
# 2. Add → PointCloud2
# 3. Set Topic to: /stereo/points
# 4. Set Color Transformer to: RGB8
# 5. Adjust point size as needed
# 6. Invert Z-axis
```

### ROS2 Topic Information

```bash
# List all topics
ros2 topic list

# Check topic info
ros2 topic info /stereo/points

# View point cloud data
ros2 topic echo /stereo/points

# Check publishing rate
ros2 topic hz /stereo/points
```

**Expected Rate**: ~30 Hz

##  Technical Specifications

### Camera Setup

| Parameter | Value | Unit | Notes |
|-----------|-------|------|-------|
| Camera Type | USB Webcams | - | Consumer-grade (UGREEN) |
| Left Camera Index | 0 or 2 | - | Detected automatically |
| Right Camera Index | 2 or 0 | - | Detected automatically |
| Resolution | 640 × 480 | pixels | Limited by hardware |
| Frame Rate | ~30 | FPS | Real-time performance |
| Baseline | 114.79 | mm | Measured from calibration |
| Focal Length | 1013.87 | pixels | After rectification |
| Shutter Type | Rolling | - | ⚠️ Can cause motion artifacts |

⚠️ **Camera Limitations**: Consumer webcams have rolling shutters, auto-exposure, and lower resolution compared to industrial cameras, which affects depth measurement consistency.

### Calibration Parameters

| Parameter | Left Camera | Right Camera |
|-----------|-------------|--------------|
| fx (focal length X) | ~1013 px | ~1013 px |
| fy (focal length Y) | ~1013 px | ~1013 px |
| cx (principal point X) | ~320 px | ~320 px |
| cy (principal point Y) | ~240 px | ~240 px |
| k1 (radial distortion) | ~-0.3 | ~-0.3 |

### Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Depth Map Coverage | 75% | With CLAHE + WLS optimization |
| Reprojection Error | 0.57 pixels | ✅ Excellent calibration quality |
| Processing Time | ~33 ms | Per frame (30 FPS) |
| Depth Range | 0.73 - 23.3 m | Practical working range |
| Optimal Range | 1.0 - 5.0 m | Best accuracy zone |
| Depth Accuracy @ 1m | ±0.9 cm | Excellent |
| Depth Accuracy @ 3m | ±7.7 cm | Good |
| Depth Accuracy @ 7m | ±36 cm | Marginal ⚠️ |
| Object Tracking FPS | ~10-15 | With YOLO inference |

⚠️ **Note on Accuracy**: While calibration is excellent (0.57px error), actual depth measurements can vary ±20-50cm at distances >5m due to camera hardware limitations, depth map quality, and environmental factors. See [Depth Estimation](#-depth-estimation) section for details.

### Algorithm Components

- **Stereo Matching**: Semi-Global Block Matching (SGBM)
- **Post-Processing**: Weighted Least Squares (WLS) filtering
- **Preprocessing**: CLAHE (Contrast Limited AHE)
- **3D Reprojection**: Q matrix transformation
- **Coordinate Frame**: Left camera optical center

##  Troubleshooting

### Issue: Objects at Same Distance Show Different Depths

**Symptoms**: Objects on the same shelf (e.g., at 7.3m) show varying depths like 3.3m, 7.3m, etc.

**Root Causes**:
1. **Depth map quality**: Some regions have valid depth pixels, others have "holes" (invalid data)
2. **Small objects**: Limited pixels with valid depth data
3. **Surface properties**: Smooth/reflective surfaces → poor stereo matching
4. **Camera limitations**: Consumer webcams with rolling shutter, auto-exposure variations

**Solutions**:
```python
# The updated depth_map_wsl.py now includes:
# 1. Multi-strategy sampling (entire object, center region, grid points)
# 2. Outlier removal using percentile filtering
# 3. Confidence indicators (High/Medium/Low)
# 4. Robust median-based depth estimation

# Press 'i' key to see confidence levels:
# [H:180px] - High confidence, many valid pixels
# [M:67px]  - Medium confidence, decent samples
# [L:23px]  - Low confidence, few pixels (shown with '?')
```

**Expected Results**: Objects on same surface should show depths within ±20-30cm of each other

**Important**: This is NOT a calibration issue (your reprojection error is excellent at 0.57px). It's a fundamental limitation of passive stereo vision with consumer cameras on textureless/reflective surfaces.

### Issue: Cameras Not Detected

**Symptoms**: "Failed to open one or both cameras" error

**Solutions**:
```bash
# 1. List video devices
ls /dev/video*

# 2. Check camera details
v4l2-ctl --list-devices

# 3. Test camera
ffplay /dev/video0

# 4. Update camera indices in code
# Edit camera IDs in Python scripts (VideoCapture(0) or VideoCapture(2))
```

### Issue: Low Depth Map Coverage (<50%)

**Symptoms**: Mostly black disparity map, few valid points

**Solutions**:
- **Improve Lighting**: Use uniform, bright lighting
2. **Add Texture**: Point cameras at textured surfaces (not blank walls)
3. **Adjust Parameters**: Increase `uniquenessRatio`, decrease `blockSize`
4. **Enable WLS**: Ensure opencv-contrib-python is installed
5. **Enable CLAHE**: Enhances local contrast

### Issue: Object Tracking Performance Slow

**Symptoms**: Low FPS when object tracking is enabled

**Solutions**:
1. **Toggle tracking off**: Press 't' key to disable when not needed
2. **Reduce confidence threshold**: Lower `CONF_THRESHOLD` to 0.3 for faster processing
3. **Use GPU acceleration**: Install CUDA-enabled PyTorch for YOLO
   ```bash
   # For NVIDIA GPU:
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
4. **Reduce resolution**: Lower camera resolution if high precision not needed

### Issue: Object Not Detected

**Symptoms**: YOLO doesn't detect certain objects

**Solutions**:
1. **Lower confidence threshold**: Edit `CONF_THRESHOLD` in code (default 0.5)
2. **Check YOLO classes**: YOLO v8 trained on 80 COCO classes
   - See: https://docs.ultralytics.com/datasets/detect/coco/
3. **Improve lighting**: Better lighting → better detection
4. **Object size**: Very small objects (<20px) may not be detected

### Issue: Incorrect Depth Values

**Symptoms**: Depth shows very large or very small values, inconsistent measurements

**Solutions**:
1. **Check if values are in millimeters**: If depth > 20m, likely in mm
   ```python
   # Auto-conversion is included in updated code
   if depth_value > 20.0:  # Likely mm
       depth_m = depth_value / 1000.0
   ```

2. **Verify baseline units**: Calibration T vector should be in mm
   ```python
   # In depth_map_wsl.py, baseline is converted:
   baseline = np.linalg.norm(T) / 1000.0  # Convert mm to meters
   ```

3. **Check object is in valid range**: 0.73m - 23.3m working range

4. **Improve measurement quality**:
   - Add more lighting (uniform, diffuse)
   - Point camera at textured surfaces
   - Avoid smooth/reflective objects
   - Use objects >15×15 pixels in size

5. **Use confidence indicators**: Press 'i' to see pixel counts
   - High confidence (>100 pixels) → Trust the measurement
   - Low confidence (<30 pixels) → Be cautious, marked with '?'

### Issue: Checkerboard Not Detected

**Symptoms**: Calibration cannot find pattern

**Solutions**:
1. **Verify Pattern Size**: Must be 8×6 internal corners
2. **Improve Lighting**: Avoid shadows and glare
3. **Check Focus**: Ensure cameras are in focus
4. **Flat Surface**: Print on rigid, flat surface
5. **Adjust Threshold**: Modify detection parameters if needed

### Issue: Point Cloud in Wrong Units

**Symptoms**: Point cloud extremely large or small

**Solutions**:
```python
# Auto-detect units in code
z_med = np.median(np.abs(Z))
units = 'mm' if z_med > 20.0 else 'm'

# Apply conversion if needed
if units == 'mm':
    points = points / 1000.0  # Convert to meters
```

### Issue: ROS2 Node Not Found

**Symptoms**: `ros2 run cam_ros_node` command fails

**Solutions**:
```bash
# 1. Rebuild package
cd ~/cam_ros_node
colcon build --packages-select cam_ros_node

# 2. Source workspace
source install/setup.bash

# 3. Verify node exists
ros2 pkg executables cam_ros_node

# 4. Check setup.py entry points
cat cam_ros_node/setup.py
```

### Issue: Poor Stereo Matching

**Symptoms**: Noisy disparity map, incorrect depths

**Solutions**:
1. **Recalibrate**: Achieve < 1.0 pixel reprojection error
2. **Check Rectification**: Epipolar lines should be horizontal
3. **Adjust SGBM**: Tune `P1`, `P2`, `uniquenessRatio`
4. **Enable WLS**: Significantly improves quality
5. **Add Texture**: Point at objects with visible texture

##  Additional Resources

### Understanding Your Depth Measurements

The depth measurement system is based on stereo triangulation:

**Quick Facts About Your Setup:**
- Baseline: 114.85mm → Good for 1-5m range
- Calibration: 0.57px error → Excellent quality
- Depth range: 0.73m - 23.3m (practical)
- Best accuracy: 1-3m (±1-8cm)
- At 7m: ±36cm accuracy (expect ±20-50cm variation with consumer cameras)

**Depth Formula**: `Z = (Focal_Length × Baseline) / Disparity`

### External Documentation
- [OpenCV Stereo Calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [SGBM Algorithm](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html)
- [Open3D Point Cloud](http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html)
- [ROS2 Humble Docs](https://docs.ros.org/en/humble/)


### Tools
- **Meshlab**: Point cloud visualization and processing
- **CloudCompare**: Advanced point cloud analysis
- **RViz**: ROS visualization tool
- **rqt**: ROS2 GUI tools

##  Contributing

Contributions are welcome! 


### Code Style
- Followed Python code
- Add docstrings to all functions
- Include comments for complex algorithms
- Test your changes before submitting

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Author

**Sourav**
- GitHub: [@Sourav0607](https://github.com/Sourav0607)
- Repository: [Stereo_Vision_ROS2](https://github.com/Sourav0607/Stereo_Vision_ROS2)

##  Acknowledgments

- OpenCV community for excellent computer vision libraries
- Open3D developers for 3D visualization tools
- ROS2 community for robotics middleware
- Stereo vision research community

##  Project Status

**Status**:  Active Development

**Completed Features**:
-  Stereo camera calibration (manual and auto-capture modes)
-  Real-time depth map generation with SGBM
-  WLS (Weighted Least Squares) filtering for disparity refinement
-  Achieved 75% depth map coverage with excellent 0.57px calibration error
-  Multi-strategy depth sampling for robust measurements
-  Real-time object detection and tracking (YOLO v8 + SORT)
-  Distance measurement for tracked objects with confidence indicators
-  Interactive depth visualization (grayscale and colored modes)
-  3D point cloud visualization with Open3D
-  Interactive mouse-click depth measurement
-  Depth analysis and statistics tools
-  ROS2 camera image publisher node
-  ROS2 stereo point cloud publisher node with PointCloud2 messages
-  RViz visualization support
-  Comprehensive documentation and troubleshooting guides
-  Comprehensive code comments

**Future Work** (Not Yet Completed):
- [ ] **Upgrade to industrial cameras** for improved consistency
  - Global shutter cameras (eliminate rolling shutter artifacts)
  - Higher resolution (1920×1080 or better)
  - Fixed exposure and white balance
- [ ] **Implement active stereo** for textureless surfaces
  - Structured light projection
  - Time-of-flight (ToF) sensor integration
- [ ] **Multi-frame depth averaging** for temporal smoothing
- [ ] **Deep learning depth estimation** as fallback
- [ ] Implement human pose estimation with depth
- [ ] Real-time SLAM integration

---

**Happy Stereo Vision!**

For questions or issues, please write me to sourav.hawaldar@gmail.com
