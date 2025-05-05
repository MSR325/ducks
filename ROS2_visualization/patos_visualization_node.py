#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import pandas as pd
import time
import os
import random
import math
from tf_transformations import quaternion_from_matrix, quaternion_from_euler
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np

# Assign unique colors to each duck


class DuckVisualizer(Node):
    def __init__(self, csv_path):
        super().__init__('duck_visualizer')
        self.marker_pub = self.create_publisher(MarkerArray, 'duck_markers', 10)
        self.path_pub = self.create_publisher(MarkerArray, 'duck_paths', 10)

        # Load duck positions
        self.df = pd.read_csv(csv_path)
        self.df.sort_values(by=['frame', 'duck_id'], inplace=True)
        self.frames = self.df['frame'].unique()
        self.current_frame_index = 0

        # Timer to update visualization
        self.timer = self.create_timer(0.033, self.timer_callback)  # 10 Hz

        # Store path history for each duck
        self.duck_paths = {}

        # Absolute path to your STL file
        self.mesh_path = "file:/home/roger/Github/rogertests/rogertests/RETO_PATOS/3D_Reconstruction/PATO_IBAN/Shaded/base.stl"

        self.duck_colors = {}
        self.prev_duck_ids = set()

        self.duck_last_position = {}  # duck_id -> (x, y)
        self.max_ducks = 7

        self.cam_c2w = np.load("/home/roger/Github/rogertests/rogertests/RETO_PATOS/video_droid.npz")["cam_c2w"]  # (N, 4, 4)
        self.tf_broadcaster = TransformBroadcaster(self)


    def generate_color(self):
        return (
            random.uniform(0.2, 1.0),
            random.uniform(0.2, 1.0),
            random.uniform(0.2, 1.0)
        )


    def compute_yaw_quaternion(self, dx, dy):
        yaw = math.atan2(dy, dx)
        q = quaternion_from_euler(0, 0, yaw)  # roll, pitch, yaw
        return q
    
    def timer_callback(self):
        if self.current_frame_index >= len(self.frames):
            return

        current_frame = self.frames[self.current_frame_index]
        self.get_logger().info(f"📸 Visualizing frame {current_frame}")

        # Use one consistent timestamp for all messages
        now = self.get_clock().now().to_msg()

        frame_df = self.df[self.df['frame'] == current_frame]
        marker_array = MarkerArray()
        path_array = MarkerArray()

        # --- Broadcast TF: map -> camera_frame ---
        # Load raw camera pose (camera-to-world)
        use_camera_frame = self.current_frame_index < len(self.cam_c2w)

        if use_camera_frame:
            T = self.cam_c2w[self.current_frame_index]

            # Convert COLMAP/DROID -> ROS frame
            convert_to_ros = np.array([
                [0,  0, 1, 0],   # X_ros = Z_colmap
                [-1, 0, 0, 0],   # Y_ros = -X_colmap
                [0, -1, 0, 0],   # Z_ros = -Y_colmap
                [0,  0, 0, 1]
            ])

            theta = 0  # -90 degrees in radians

            rot_y = np.array([
                [np.cos(theta), 0, np.sin(theta), 0],
                [0,             1, 0,             0],
                [-np.sin(theta),0, np.cos(theta), 0],
                [0,             0, 0,             1]
            ])

            # Correct order: convert frame first, rotate, then lift
            T = T @ convert_to_ros
            T = T @ rot_y
            T[:3, 3] += np.array([0.0, 0.0, 1.5])  # Lift in ROS Z-axis



            translation = T[:3, 3]
            rotation_matrix = T[:3, :3]
            yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            q = quaternion_from_matrix(T)  # This takes the full 4x4 matrix

            tf_msg = TransformStamped()
            tf_msg.header.stamp = now
            tf_msg.header.frame_id = "map"
            tf_msg.child_frame_id = "camera_frame"
            tf_msg.transform.translation.x = float(translation[0])
            tf_msg.transform.translation.y = float(translation[1])
            tf_msg.transform.translation.z = float(translation[2])
            tf_msg.transform.rotation.x = float(q[0])
            tf_msg.transform.rotation.y = float(q[1])
            tf_msg.transform.rotation.z = float(q[2])
            tf_msg.transform.rotation.w = float(q[3])
            self.tf_broadcaster.sendTransform(tf_msg)
            

            T_world_cam = np.linalg.inv(T)

        # --- Update last known positions for detected ducks ---
        for _, row in frame_df.iterrows():
            duck_id = int(row['duck_id'])
            x = float(row['x_world'])
            y = float(row['y_world'])
            self.duck_last_position[duck_id] = (x, y)

        # --- Publish all 7 duck markers ---
        for duck_id in range(self.max_ducks):
            if duck_id not in self.duck_last_position:
                continue

            x_local, y_local = self.duck_last_position[duck_id]

            if self.current_frame_index < len(self.cam_c2w):
                camera_x, camera_y = translation[0], translation[1]
                x = x_local + camera_x
                y = y_local + camera_y
            else:
                x = x_local
                y = y_local

            z = 0.0

            # Initialize path if needed
            if duck_id not in self.duck_paths:
                self.duck_paths[duck_id] = []

            # Only add to path if moved
            last = self.duck_paths[duck_id][-1] if self.duck_paths[duck_id] else None
            if not last or last.x != x or last.y != y:
                self.duck_paths[duck_id].append(Point(x=x, y=y, z=z))

            # Orientation based on movement
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = now
            marker.ns = "ducks"
            marker.id = duck_id
            marker.type = Marker.MESH_RESOURCE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z

            if len(self.duck_paths[duck_id]) >= 2:
                prev = self.duck_paths[duck_id][-2]
                dx = x - prev.x
                dy = y - prev.y
                if dx != 0 or dy != 0:
                    q = self.compute_yaw_quaternion(dx, dy)
                    marker.pose.orientation.x = q[0]
                    marker.pose.orientation.y = q[1]
                    marker.pose.orientation.z = q[2]
                    marker.pose.orientation.w = q[3]
                else:
                    marker.pose.orientation.w = 1.0
            else:
                marker.pose.orientation.w = 1.0

            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.mesh_resource = self.mesh_path
            marker.mesh_use_embedded_materials = False
            marker.color.r = 1.0
            marker.color.g = 0.85
            marker.color.b = 0.1
            marker.color.a = 1.0
            marker_array.markers.append(marker)

            # Assign unique color
            if duck_id not in self.duck_colors:
                self.duck_colors[duck_id] = self.generate_color()
            r, g, b = self.duck_colors[duck_id]

            # Path line marker
            path_marker = Marker()
            path_marker.header.frame_id = "map"
            path_marker.header.stamp = now
            path_marker.ns = f"path_{duck_id}"
            path_marker.id = 1000 + duck_id
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            path_marker.scale.x = 0.01
            path_marker.color.r = r
            path_marker.color.g = g
            path_marker.color.b = b
            path_marker.color.a = 1.0
            path_marker.points = self.duck_paths[duck_id]
            path_array.markers.append(path_marker)

        self.marker_pub.publish(marker_array)
        self.path_pub.publish(path_array)
        self.current_frame_index += 1


def main():
    rclpy.init()
    node = DuckVisualizer("/home/roger/Github/rogertests/rogertests/RETO_PATOS/3D_Reconstruction/duck_world_positions_lalooooo.csv")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
