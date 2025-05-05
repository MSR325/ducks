import cv2
import numpy as np

# ----------- SETTINGS -----------
IMG_PATHS = ['video_imgs/frame_00000.jpg', 'video_imgs/frame_00050.jpg']
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

def select_points(image, n=4):
    points = []
    
    temp = image.copy()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n:
            points.append((x, y))
            cv2.circle(temp, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select 4 Tile Corners", temp)

    cv2.imshow("Select 4 Tile Corners", temp)
    cv2.setMouseCallback("Select 4 Tile Corners", click_event)

    while len(points) < n:
        cv2.waitKey(10)

    cv2.destroyAllWindows()
    return np.array(points, dtype=np.float32)

# Define the canonical (ideal) square
canonical_tile = np.array([
    [0, 0],
    [TILE_SIZE_PX, 0],
    [TILE_SIZE_PX, TILE_SIZE_PX],
    [0, TILE_SIZE_PX]
], dtype=np.float32)
canonical_tile += np.array(CENTER_POS) - TILE_SIZE_PX // 2

# Process all images
warped_imgs = []
offsets = []
all_points = []
all_transforms = []

for path in IMG_PATHS:
    img = cv2.imread(path)
    print(f"[INFO] Select the same tile in image: {path}")
    tile_pts = select_points(img)
    all_points.append(tile_pts)

    H, _ = cv2.findHomography(tile_pts, canonical_tile)
    warped, full_H, offset = warp_image_with_full_view(img, H)

    warped_imgs.append((warped, offset))
    all_transforms.append((full_H, offset))

# Determine output canvas bounds
all_xmin = min(offset[0] for _, offset in warped_imgs)
all_ymin = min(offset[1] for _, offset in warped_imgs)
all_xmax = max(offset[0] + img.shape[1] for img, offset in warped_imgs)
all_ymax = max(offset[1] + img.shape[0] for img, offset in warped_imgs)

canvas_w = all_xmax - all_xmin
canvas_h = all_ymax - all_ymin
mosaic = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

# Blend warped images onto mosaic
for img, (ox, oy) in warped_imgs:
    x0 = ox - all_xmin
    y0 = oy - all_ymin
    h, w = img.shape[:2]
    mask = img > 0
    mosaic[y0:y0+h, x0:x0+w][mask] = img[mask]

# Draw points from each warped image
colors = [(0, 0, 255), (0, 255, 0)]  # red for image 1, green for image 2
for i, (pts, (H, offset)) in enumerate(zip(all_points, all_transforms)):
    warped_pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    for pt in warped_pts:
        px = int(pt[0] + offset[0] - all_xmin)
        py = int(pt[1] + offset[1] - all_ymin)
        cv2.circle(mosaic, (px, py), 6, colors[i], -1)

# Resize mosaic to screen-fit
screen_width, screen_height = 1920, 1080
mh, mw = mosaic.shape[:2]
scale = min(screen_width / mw, screen_height / mh, 1.0)
resized = cv2.resize(mosaic, (int(mw * scale), int(mh * scale)))

# Show result
cv2.namedWindow("Rectified Floor Mosaic", cv2.WINDOW_NORMAL)
cv2.imshow("Rectified Floor Mosaic", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save output
cv2.imwrite("3D_Reconstruction/floor_reconstruct_attempts/floor_rectify_attempt_results/floor_rectified_mosaic_fixed.png", mosaic)
