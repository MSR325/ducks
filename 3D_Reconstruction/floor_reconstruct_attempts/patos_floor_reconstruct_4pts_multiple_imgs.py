import cv2
import numpy as np

# ----------- SETTINGS -----------
IMG_PATHS = [
    'video_imgs/frame_00000.jpg',
    'video_imgs/frame_00050.jpg',
    'video_imgs/frame_00100.jpg'
]
TILE_SIZE_PX = 200
CENTER_POS = (1000, 1000)
# --------------------------------

def warp_image_with_full_view(img, H):
    h, w = img.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    [xmin, ymin] = np.floor(warped_corners.min(axis=0)[0]).astype(int)
    [xmax, ymax] = np.ceil(warped_corners.max(axis=0)[0]).astype(int)
    width, height = xmax - xmin, ymax - ymin

    T = np.array([
        [1, 0, -xmin],
        [0, 1, -ymin],
        [0, 0, 1]
    ], dtype=np.float64)

    full_H = T @ H
    warped = cv2.warpPerspective(img, full_H, (width, height))
    return warped, full_H, (xmin, ymin)

def select_points(image, n=4, color=(0, 0, 255)):
    points = []
    temp = image.copy()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n:
            points.append((x, y))
            cv2.circle(temp, (x, y), 5, color, -1)
            cv2.imshow("Select 4 Tile Corners", temp)

    cv2.imshow("Select 4 Tile Corners", temp)
    cv2.setMouseCallback("Select 4 Tile Corners", click_event)

    while len(points) < n:
        cv2.waitKey(10)

    cv2.destroyAllWindows()
    return np.array(points, dtype=np.float32)

# Canonical tile
canonical_tile = np.array([
    [0, 0],
    [TILE_SIZE_PX, 0],
    [TILE_SIZE_PX, TILE_SIZE_PX],
    [0, TILE_SIZE_PX]
], dtype=np.float32)
canonical_tile += np.array(CENTER_POS) - TILE_SIZE_PX // 2

# Load images
img1 = cv2.imread(IMG_PATHS[0])
img2 = cv2.imread(IMG_PATHS[1])
img3 = cv2.imread(IMG_PATHS[2])

# Align image 1 to canonical
print(f"[INFO] Select tile in: {IMG_PATHS[0]}")
pts1 = select_points(img1, color=(0, 0, 255))
H1, _ = cv2.findHomography(pts1, canonical_tile)
warped1, H1_full, offset1 = warp_image_with_full_view(img1, H1)

# Align image 2 to image 1
print(f"[INFO] Select corresponding tile in: {IMG_PATHS[1]} (same tile as in image 1)")
pts2 = select_points(img2, color=(0, 255, 0))
H2_1, _ = cv2.findHomography(pts2, pts1)  # image2 to image1
H2 = H1 @ H2_1  # image2 to canonical
warped2, H2_full, offset2 = warp_image_with_full_view(img2, H2)

# Align image 3 to image 2
print(f"[INFO] Select corresponding tile in: {IMG_PATHS[2]} (same tile as in image 2)")
pts3 = select_points(img3, color=(255, 0, 0))
H3_2, _ = cv2.findHomography(pts3, pts2)  # image3 to image2
H3 = H2 @ H3_2  # image3 to canonical
warped3, H3_full, offset3 = warp_image_with_full_view(img3, H3)

# Build mosaic canvas
all_offsets = [offset1, offset2, offset3]
all_warped = [warped1, warped2, warped3]

xmin = min(o[0] for o in all_offsets)
ymin = min(o[1] for o in all_offsets)
xmax = max(o[0] + img.shape[1] for img, o in zip(all_warped, all_offsets))
ymax = max(o[1] + img.shape[0] for img, o in zip(all_warped, all_offsets))

canvas_w = xmax - xmin
canvas_h = ymax - ymin
mosaic = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

# Blend images into mosaic
for warped, offset in zip(all_warped, all_offsets):
    x0, y0 = offset[0] - xmin, offset[1] - ymin
    h, w = warped.shape[:2]
    mask = warped > 0
    mosaic[y0:y0+h, x0:x0+w][mask] = warped[mask]

# Resize and display
screen_width, screen_height = 1920, 1080
mh, mw = mosaic.shape[:2]
scale = min(screen_width / mw, screen_height / mh, 1.0)
resized = cv2.resize(mosaic, (int(mw * scale), int(mh * scale)))

cv2.namedWindow("Rectified Floor Mosaic", cv2.WINDOW_NORMAL)
cv2.imshow("Rectified Floor Mosaic", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("3D_Reconstruction/floor_reconstruct_attempts/floor_rectify_attempt_results/floor_rectified_mosaic_3imgs.png", mosaic)
