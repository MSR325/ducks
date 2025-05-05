import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "3D_Reconstruction/room_reconstruct/test_imgs/striped.jpg"
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale and detect edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 20, 120, apertureSize=3)

# Hough Transform to detect lines
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10)

# Prepare image copies
img_lines = img_rgb.copy()
img_horizontal = img_rgb.copy()
img_vertical = img_rgb.copy()
img_left_wall = img_rgb.copy()
img_right_wall = img_rgb.copy()

# Image dimensions
height, width = img.shape[:2]
left_wall_x_range = (0, width * 0.5)
right_wall_x_range = (width * 0.5, width)
bottom_exclusion_zone = height * 0.75

# Angle thresholds
horizontal_thresh = 3
vertical_thresh = 15
oblique_thresh_min = 20
oblique_thresh_max = 70

# Process lines
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        avg_y = (y1 + y2) / 2
        avg_x = (x1 + x2) / 2

        # All lines
        cv2.line(img_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Horizontal lines
        if angle < horizontal_thresh or angle > (180 - horizontal_thresh):
            cv2.line(img_horizontal, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Vertical lines
        elif abs(angle - 90) < vertical_thresh:
            cv2.line(img_vertical, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Left wall lines (oblique + left side + not on floor)
        if (oblique_thresh_min < angle < oblique_thresh_max or
            (180 - oblique_thresh_max) < angle < (180 - oblique_thresh_min)):
            if left_wall_x_range[0] <= avg_x <= left_wall_x_range[1] and avg_y < bottom_exclusion_zone:
                cv2.line(img_left_wall, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # Right wall lines (oblique + right side + not on floor)
        if (oblique_thresh_min < angle < oblique_thresh_max or
            (180 - oblique_thresh_max) < angle < (180 - oblique_thresh_min)):
            if right_wall_x_range[0] <= avg_x <= right_wall_x_range[1] and avg_y < bottom_exclusion_zone:
                cv2.line(img_right_wall, (x1, y1), (x2, y2), (0, 255, 255), 2)

# Plot results
fig, axs = plt.subplots(2, 4, figsize=(22, 10))

axs[0, 0].imshow(img_rgb)
axs[0, 0].set_title("Original Room Image")
axs[0, 0].axis("off")

axs[0, 1].imshow(edges, cmap='gray')
axs[0, 1].set_title("Canny Edges")
axs[0, 1].axis("off")

axs[0, 2].imshow(img_lines)
axs[0, 2].set_title("All Hough Lines (Red)")
axs[0, 2].axis("off")

axs[0, 3].imshow(img_horizontal)
axs[0, 3].set_title("Horizontal Lines (Green)")
axs[0, 3].axis("off")

axs[1, 0].imshow(img_vertical)
axs[1, 0].set_title("Vertical Lines (Blue)")
axs[1, 0].axis("off")

axs[1, 1].imshow(img_left_wall)
axs[1, 1].set_title("Left Wall Lines (Magenta)")
axs[1, 1].axis("off")

axs[1, 2].imshow(img_right_wall)
axs[1, 2].set_title("Right Wall Lines (Cyan)")
axs[1, 2].axis("off")

axs[1, 3].axis("off")  # Hide the empty subplot

plt.tight_layout()
plt.show()
