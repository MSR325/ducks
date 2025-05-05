import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def detect_tile_corners_via_hough(gray):
    """Detects tile grid intersections (tile corners) using HoughLinesP"""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=40, maxLineGap=5)

    horizontal = []
    vertical = []

    if lines is None:
        return []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if abs(angle) < 10:
            horizontal.append((x1, y1, x2, y2))
        elif abs(angle - 90) < 10 or abs(angle + 90) < 10:
            vertical.append((x1, y1, x2, y2))

    # Get unique x and y positions from lines to compute intersections
    def line_midpoints(lines):
        return [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2 in lines]

    h_points = line_midpoints(horizontal)
    v_points = line_midpoints(vertical)

    # Cartesian product of horizontal and vertical lines gives intersection guesses
    intersections = []
    for hx, hy in h_points:
        for vx, vy in v_points:
            intersections.append((vx, hy))

    return intersections


def draw_detected_points(image, points, radius=4, color=(0, 0, 255)):
    out = image.copy()
    for (x, y) in points:
        cv2.circle(out, (x, y), radius, color, -1)
    return out


# Load and test on a single frame
test_img_path = "video_imgs/frame_00000.jpg"
image = cv2.imread(test_img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detected_points = detect_tile_corners_via_hough(gray)
image_with_points = draw_detected_points(image, detected_points)

# Display using matplotlib (so it works in this environment)
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB))
plt.title("Detected Tile Grid Intersections via HoughLines")
plt.axis("off")
plt.show()
