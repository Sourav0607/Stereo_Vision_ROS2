#!/usr/bin/env python3
"""
ROS2 Stereo Camera Publisher Node

This node captures frames from two USB cameras (left and right) and publishes
them as ROS2 Image messages on separate topics for stereo vision processing.

Topics Published:
- /camera/left/image_raw (sensor_msgs/Image): Left camera images at ~30 FPS
- /camera/right/image_raw (sensor_msgs/Image): Right camera images at ~30 FPS

Usage:
    ros2 run cam_ros_node cam_ros_node

Requirements:
- rclpy (ROS2 Python client library)
- sensor_msgs (ROS2 message definitions)
- cv_bridge (OpenCV <-> ROS message converter)
- opencv-python (camera capture and image processing)

Author: Sourav Hawaldar
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class StereoCameraPublisher(Node):
    """
    ROS2 Node for publishing stereo camera image streams
    
    This class inherits from rclpy.node.Node and handles:
    1. Opening two USB cameras (left and right)
    2. Capturing frames continuously
    3. Converting OpenCV images to ROS2 Image messages
    4. Publishing images on separate topics at ~30 FPS
    """
    def __init__(self):
        # Initialize the ROS2 node with name 'cam_ros_node'
        # This name will appear in `ros2 node list`
        super().__init__('cam_ros_node')

        # === Create ROS2 Publishers ===
        # Publishers send messages to specific topics
        # Other nodes can subscribe to these topics to receive images
        # Queue size = 10: keeps last 10 messages if subscriber is slow
        self.left_pub = self.create_publisher(Image, '/camera/left/image_raw', 10)
        self.right_pub = self.create_publisher(Image, '/camera/right/image_raw', 10)

        # === Initialize CvBridge ===
        # CvBridge converts between OpenCV images (numpy arrays) and ROS Image messages
        # This allows OpenCV-captured images to be published on ROS topics
        self.bridge = CvBridge()

        # === Open USB Cameras ===
        # Camera indices depend on your system's /dev/video* enumeration
        # Change video index if required (check with `v4l2-ctl --list-devices`)
        # Index 2: Left camera (matches stereo calibration setup)
        # Index 0: Right camera (matches stereo calibration setup)
        self.left_cam = cv2.VideoCapture(2)
        self.right_cam = cv2.VideoCapture(0)

        # Verify both cameras opened successfully
        # isOpened() returns True if camera device is accessible
        if not self.left_cam.isOpened() or not self.right_cam.isOpened():
            self.get_logger().error("Failed to open one or both cameras!")
            exit(1)

        # === Create Timer for Frame Publishing ===
        # Timer at 30 FPS (approx): 1/30 = 0.033 seconds per frame
        # Callback function publish_frames() will be called every 33ms
        # This creates a periodic publishing loop without blocking
        self.timer = self.create_timer(0.033, self.publish_frames)

        # Log successful initialization to ROS2 console
        # Visible with `ros2 topic echo /rosout` or in rqt_console
        self.get_logger().info("Stereo camera publisher started.")

    def publish_frames(self):
        """
        Timer callback function: captures and publishes camera frames
        
        This function is called every 33ms (30 FPS) by the ROS2 timer.
        It performs:
        1. Capture frames from both cameras
        2. Convert OpenCV BGR images to ROS Image messages
        3. Publish messages to their respective topics
        
        Note: Cameras are read independently so one failing doesn't block the other
        """
        # === Capture Left Camera Frame ===
        # read() returns (success_flag, image_array)
        # ret_left: True if frame captured successfully, False otherwise
        # frame_left: numpy array (HxWx3) in BGR color format
        ret_left, frame_left = self.left_cam.read()
        # === Capture Right Camera Frame ===
        ret_right, frame_right = self.right_cam.read()

        # === Publish Left Camera Image ===
        if ret_left:
            # Convert OpenCV image (numpy BGR array) to ROS Image message
            # encoding='bgr8': 8-bit BGR color format (OpenCV default)
            # Other encodings: 'rgb8', 'mono8', 'rgba8', etc.
            img_msg = self.bridge.cv2_to_imgmsg(frame_left, encoding='bgr8')
            # Publish message to /camera/left/image_raw topic
            # Subscribers can receive this with `ros2 topic echo` or in their nodes
            self.left_pub.publish(img_msg)

        # === Publish Right Camera Image ===
        if ret_right:
            # Same conversion and publishing process for right camera
            img_msg = self.bridge.cv2_to_imgmsg(frame_right, encoding='bgr8')
            # Publish message to /camera/right/image_raw topic
            self.right_pub.publish(img_msg)

    def destroy_node(self):
        """
        Cleanup function called when node is shutting down
        
        This ensures proper resource cleanup:
        1. Release camera devices (frees /dev/video* for other applications)
        2. Call parent class cleanup (closes publishers, timers, etc.)
        
        Called automatically when node exits or receives shutdown signal
        """
        # Release camera resources to free hardware devices
        # Important: prevents "device busy" errors for other applications
        self.left_cam.release()
        self.right_cam.release()
        # Call parent class destroy_node to clean up ROS2 resources
        super().destroy_node()

def main(args=None):
    """
    Main entry point for the ROS2 node
    
    This function:
    1. Initializes the ROS2 Python client library (rclpy)
    2. Creates the camera publisher node
    3. Spins (runs) the node until interrupted
    4. Handles cleanup on exit
    
    Args:
        args: Command-line arguments (optional, passed from ROS2 launch)
    
    Usage:
        ros2 run cam_ros_node cam_ros_node
    """
    # Initialize ROS2 communications
    # This sets up context for creating nodes, publishers, subscribers, etc.
    rclpy.init(args=args)
    
    # Create instance of StereoCameraPublisher node
    # This initializes cameras, publishers, and starts the timer
    node = StereoCameraPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
