import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, Frame, Button, Label, Scale, HORIZONTAL
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

class RGBDViewer:
    def __init__(self, root, npz_path):
        self.root = root
        self.root.title("Cute RGB-D Viewer")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f8ff")
        
        # Load data from NPZ file
        print("Loading data...")
        self.data = np.load(npz_path)
        self.images = self.data['images']  # RGB frames
        self.depths = self.data['depths']  # Depth maps
        self.intrinsic = self.data['intrinsic']  # Camera intrinsic parameters
        self.cam_poses = self.data['cam_c2w']  # Camera to world transformations
        print("Data loaded successfully")
        
        self.num_frames = len(self.images)
        self.current_frame = 0
        self.is_playing = False
        self.play_speed = 50  # milliseconds between frames
        
        # Set up the UI components
        self.setup_ui()
        
        # Update display with first frame
        self.root.after(100, self.update_display)
    
    def setup_ui(self):
        # Title with cute font
        title_label = Label(self.root, text="✨ RGB-D Viewer with Camera Pose ✨", 
                          font=('Arial', 16, 'bold'), bg="#f0f8ff", fg="#5e72e4")
        title_label.pack(pady=10)
        
        # Create a main frame for the top row (RGB, Depth, Trajectory)
        main_frame = Frame(self.root, bg="#f0f8ff")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left frame for RGB and Depth
        display_frame = Frame(main_frame, bg="#f0f8ff")
        display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # RGB frame
        rgb_container = Frame(display_frame, bg="white", bd=2, relief="raised")
        rgb_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        rgb_title = Label(rgb_container, text="RGB View", 
                        font=('Arial', 12, 'bold'), bg="white", fg="#5e72e4")
        rgb_title.pack(pady=5)
        
        self.rgb_label = Label(rgb_container, bg="white")
        self.rgb_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Depth frame
        depth_container = Frame(display_frame, bg="white", bd=2, relief="raised")
        depth_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        depth_title = Label(depth_container, text="Depth View", 
                          font=('Arial', 12, 'bold'), bg="white", fg="#5e72e4")
        depth_title.pack(pady=5)
        
        self.depth_label = Label(depth_container, bg="white")
        self.depth_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right frame for camera trajectory
        trajectory_container = Frame(main_frame, bg="white", bd=2, relief="raised")
        trajectory_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        trajectory_title = Label(trajectory_container, text="Camera Trajectory", 
                              font=('Arial', 12, 'bold'), bg="white", fg="#5e72e4")
        trajectory_title.pack(pady=5)
        
        # Create a matplotlib figure for the trajectory
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=trajectory_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize the trajectory plot
        self.plot_trajectory()
        
        # Controls area
        controls_frame = Frame(self.root, bg="white", bd=2, relief="raised")
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Frame slider
        slider_frame = Frame(controls_frame, bg="white")
        slider_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.frame_slider = Scale(slider_frame, from_=0, to=self.num_frames-1, 
                                orient=HORIZONTAL, command=self.slider_changed,
                                bg="white", highlightthickness=0, troughcolor="#dce4f4")
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.frame_counter = Label(slider_frame, text=f"Frame: 0/{self.num_frames-1}", 
                                 bg="white", fg="#5e72e4", font=('Arial', 10))
        self.frame_counter.pack(side=tk.RIGHT, padx=10)
        
        # Playback buttons
        buttons_frame = Frame(controls_frame, bg="white")
        buttons_frame.pack(pady=5)
        
        # Configure the cute button style
        button_style = {'font': ('Arial', 10, 'bold'), 
                      'bg': '#8ec5fc', 
                      'fg': 'white', 
                      'width': 3, 
                      'height': 1,
                      'bd': 0,
                      'relief': 'flat',
                      'padx': 10}
        
        # Add rounded corners with a frame trick
        button_frame_style = {'bg': 'white', 'padx': 5}
        
        prev_frame = Frame(buttons_frame, **button_frame_style)
        prev_frame.pack(side=tk.LEFT, padx=5)
        self.prev_button = Button(prev_frame, text="start", **button_style, command=self.prev_frame)
        self.prev_button.pack()
        
        play_frame = Frame(buttons_frame, **button_frame_style)
        play_frame.pack(side=tk.LEFT, padx=5)
        self.play_button = Button(play_frame, text="play", **button_style, command=self.toggle_play)
        self.play_button.pack()
        
        next_frame = Frame(buttons_frame, **button_frame_style)
        next_frame.pack(side=tk.LEFT, padx=5)
        self.next_button = Button(next_frame, text="end", **button_style, command=self.next_frame)
        self.next_button.pack()
        
        # Speed control
        speed_label = Label(buttons_frame, text="Speed:", bg="white", fg="#5e72e4", font=('Arial', 10))
        speed_label.pack(side=tk.LEFT, padx=(20, 5))
        
        self.speed_slider = Scale(buttons_frame, from_=10, to=200, orient=HORIZONTAL,
                                command=self.update_speed, bg="white", length=100,
                                highlightthickness=0, troughcolor="#dce4f4")
        self.speed_slider.set(50)
        self.speed_slider.pack(side=tk.LEFT)
    
    def plot_trajectory(self):
        """Plot the camera trajectory as a 3D path with matplotlib"""
        # Clear previous plot
        self.ax.clear()
        
        # Extract camera positions from transformation matrices
        positions = np.array([pose[:3, 3] for pose in self.cam_poses])
        
        # Plot the trajectory
        self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=1)
        
        # Plot all camera positions
        self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', s=5)
        
        # Plot the current camera position
        current_pos = self.cam_poses[self.current_frame][:3, 3]
        self.current_point = self.ax.scatter([current_pos[0]], [current_pos[1]], [current_pos[2]], 
                                           c='r', s=50, marker='o')
        
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Camera Path')
        
        # Set axes equal for proper 3D visualization
        self.ax.set_box_aspect([1, 1, 1])
        
        # Draw the updated plot
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_trajectory(self):
        """Update just the current camera position in the trajectory plot"""
        # Clear previous current point
        if hasattr(self, 'current_point'):
            self.current_point.remove()
        
        # Plot the new current camera position
        current_pos = self.cam_poses[self.current_frame][:3, 3]
        self.current_point = self.ax.scatter([current_pos[0]], [current_pos[1]], [current_pos[2]], 
                                           c='r', s=50, marker='o')
        
        # Draw the updated plot
        self.canvas.draw()
    
    def update_display(self):
        # Get the display dimensions once
        rgb_width = self.rgb_label.winfo_width() or 400  # Fallback value if not yet available
        rgb_height = self.rgb_label.winfo_height() or 300
        depth_width = self.depth_label.winfo_width() or 400
        depth_height = self.depth_label.winfo_height() or 300
        
        # Fixed max dimensions to prevent growing
        max_width = 640
        max_height = 480
        
        # Use the smaller of the actual or max dimensions
        rgb_width = min(rgb_width, max_width)
        rgb_height = min(rgb_height, max_height)
        depth_width = min(depth_width, max_width)
        depth_height = min(depth_height, max_height)
        
        # Update RGB image
        rgb_img = self.images[self.current_frame]
        rgb_img = Image.fromarray(rgb_img)
        
        # Calculate new dimensions while preserving aspect ratio
        rgb_img_width, rgb_img_height = rgb_img.size
        aspect_ratio = rgb_img_width / rgb_img_height
        
        if rgb_img_width > rgb_img_height:
            new_width = min(rgb_width, rgb_img_width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(rgb_height, rgb_img_height)
            new_width = int(new_height * aspect_ratio)
            
        # Resize using a more controlled approach
        rgb_img = rgb_img.resize((new_width, new_height), Image.LANCZOS)
        
        rgb_photo = ImageTk.PhotoImage(rgb_img)
        self.rgb_label.config(image=rgb_photo)
        self.rgb_label.image = rgb_photo  # Keep a reference to prevent garbage collection
        
        # Update depth image (normalize for visualization)
        depth_img = self.depths[self.current_frame]
        depth_normalized = (depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img)) * 255
        
        # Calculate new dimensions while preserving aspect ratio for depth image
        depth_img_width, depth_img_height = depth_img.shape[1], depth_img.shape[0]
        aspect_ratio = depth_img_width / depth_img_height
        
        if depth_img_width > depth_img_height:
            new_width = min(depth_width, depth_img_width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(depth_height, depth_img_height)
            new_width = int(new_height * aspect_ratio)
        
        # Create colorful depth map (rainbow colormap)
        depth_colored = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)
        
        # Apply a rainbow colormap
        # Blue -> Cyan -> Green -> Yellow -> Red -> Magenta
        
        # First transition: blue to cyan (increase G)
        mask1 = depth_normalized < 51
        depth_colored[mask1, 0] = 0  # R
        depth_colored[mask1, 1] = (depth_normalized[mask1] * 5).astype(np.uint8)  # G increases
        depth_colored[mask1, 2] = 255  # B constant
        
        # Second transition: cyan to green (decrease B)
        mask2 = (depth_normalized >= 51) & (depth_normalized < 102)
        depth_colored[mask2, 0] = 0  # R
        depth_colored[mask2, 1] = 255  # G constant
        depth_colored[mask2, 2] = (255 - (depth_normalized[mask2] - 51) * 5).astype(np.uint8)  # B decreases
        
        # Third transition: green to yellow (increase R)
        mask3 = (depth_normalized >= 102) & (depth_normalized < 153)
        depth_colored[mask3, 0] = ((depth_normalized[mask3] - 102) * 5).astype(np.uint8)  # R increases
        depth_colored[mask3, 1] = 255  # G constant
        depth_colored[mask3, 2] = 0  # B
        
        # Fourth transition: yellow to red (decrease G)
        mask4 = (depth_normalized >= 153) & (depth_normalized < 204)
        depth_colored[mask4, 0] = 255  # R constant
        depth_colored[mask4, 1] = (255 - (depth_normalized[mask4] - 153) * 5).astype(np.uint8)  # G decreases
        depth_colored[mask4, 2] = 0  # B
        
        # Fifth transition: red to magenta (increase B)
        mask5 = depth_normalized >= 204
        depth_colored[mask5, 0] = 255  # R constant
        depth_colored[mask5, 1] = 0  # G
        depth_colored[mask5, 2] = ((depth_normalized[mask5] - 204) * 5).astype(np.uint8)  # B increases
        
        depth_img = Image.fromarray(depth_colored)
        
        # Resize using a more controlled approach
        depth_img = depth_img.resize((new_width, new_height), Image.LANCZOS)
        
        depth_photo = ImageTk.PhotoImage(depth_img)
        self.depth_label.config(image=depth_photo)
        self.depth_label.image = depth_photo  # Keep a reference to prevent garbage collection
        
        # Update frame counter
        self.frame_counter.config(text=f"Frame: {self.current_frame}/{self.num_frames-1}")
        
        # Update slider position without triggering callback
        self.frame_slider.set(self.current_frame)
        
        # Update camera position in trajectory plot
        self.update_trajectory()
    
    def next_frame(self):
        self.current_frame = (self.current_frame + 1) % self.num_frames
        self.update_display()
    
    def prev_frame(self):
        self.current_frame = (self.current_frame - 1) % self.num_frames
        self.update_display()
    
    def toggle_play(self):
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_button.config(text="pause")
            self.play_animation()
        else:
            self.play_button.config(text="play")
    
    def play_animation(self):
        if self.is_playing:
            self.next_frame()
            self.root.after(self.play_speed, self.play_animation)
    
    def slider_changed(self, value):
        self.current_frame = int(float(value))
        self.update_display()
    
    def update_speed(self, value):
        self.play_speed = int(float(value))

def main():
    npz_path = "video_droid.npz"  # Default path, can be changed to use file dialog
    
    root = tk.Tk()
    app = RGBDViewer(root, npz_path)
    
    root.mainloop()

if __name__ == "__main__":
    main()
