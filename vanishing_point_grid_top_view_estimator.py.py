import cv2
import numpy as np
import matplotlib.pyplot as plt

VIDEO = "patos.mp4"

prev_vp_left = None
prev_vp_right = None
SMOOTH_ALPHA = 0.8  # 0 = no memory, 1 = infinite memory

plt.ion()
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("Line orientations")
ax.set_xlabel("cos(2θ)")
ax.set_ylabel("sin(2θ)")

def mask_green_regions(frame, sensitivity=40):
    """Create a mask to exclude green regions from processing."""
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for green color in HSV
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    # Create a mask for green regions
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Dilate the mask to ensure complete coverage of green areas
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.dilate(green_mask, kernel, iterations=1)
    
    # Invert mask to get non-green regions
    non_green_mask = cv2.bitwise_not(green_mask)
    
    # Apply the mask to the original frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=non_green_mask)
    
    return masked_frame

def detect_hough_lines(frame, canny_low=50, canny_high=150, hough_threshold=100, min_line_length=40, max_line_gap=15):
    """
    Detect lines using Hough transform with configurable parameters.
    Returns both raw lines and visualization image.
    """
    masked_frame = mask_green_regions(frame)
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
    
    # Create visualization of just the lines
    lines_vis = frame.copy()
    
    # Detect lines
    raw = cv2.HoughLinesP(edges,
                          rho=1,
                          theta=np.pi/180,
                          threshold=hough_threshold,
                          minLineLength=min_line_length,
                          maxLineGap=max_line_gap)
    
    # Draw lines on visualization image
    if raw is not None:
        for line in raw[:, 0]:
            x1, y1, x2, y2 = line
            cv2.line(lines_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red lines
    
    # Create a combined visualization with edges and lines
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    debug_vis = np.vstack([lines_vis, edges_color])
    
    return raw[:,0] if raw is not None else [], lines_vis, edges, debug_vis

def get_consistent_vps(raw_vp1, raw_vp2, image_shape):
    """Maintain consistent vanishing points across frames."""
    global prev_vp_left, prev_vp_right, SMOOTH_ALPHA

    h, w = image_shape[:2]

    # Gather any that look "in-bounds"
    candidates = []
    for vp in (raw_vp1, raw_vp2):
        if vp is None:
            continue
        x, y = vp
        if abs(x) < 10*w and abs(y) < 10*h:
            candidates.append((float(x), float(y)))

    # If we don't have two fresh points, reuse the last
    if len(candidates) < 2:
        return prev_vp_left, prev_vp_right

    # For consistent assignment, use previous VPs as reference if available
    if prev_vp_left is not None and prev_vp_right is not None:
        # Calculate distances to previous VPs
        dists_to_prev_left = [
            (x - prev_vp_left[0])**2 + (y - prev_vp_left[1])**2 
            for x, y in candidates
        ]
        dists_to_prev_right = [
            (x - prev_vp_right[0])**2 + (y - prev_vp_right[1])**2 
            for x, y in candidates
        ]
        
        # Assign candidates to minimize distance to previous VPs
        if dists_to_prev_left[0] + dists_to_prev_right[1] < dists_to_prev_left[1] + dists_to_prev_right[0]:
            vp_left, vp_right = candidates[0], candidates[1]
        else:
            vp_left, vp_right = candidates[1], candidates[0]
    else:
        # On first detection, sort by X coordinate
        vp_left, vp_right = sorted(candidates, key=lambda p: p[0])

    # On first valid pair, just take it outright
    if prev_vp_left is None or prev_vp_right is None:
        prev_vp_left, prev_vp_right = vp_left, vp_right
        return vp_left, vp_right

    # Otherwise smooth against the previous frame
    def blend(new, old):
        return (
            SMOOTH_ALPHA * old[0] + (1 - SMOOTH_ALPHA) * new[0],
            SMOOTH_ALPHA * old[1] + (1 - SMOOTH_ALPHA) * new[1]
        )

    vp_left = blend(vp_left, prev_vp_left)
    vp_right = blend(vp_right, prev_vp_right)
    prev_vp_left, prev_vp_right = vp_left, vp_right

    return vp_left, vp_right

def find_intersection(line1, line2):
    """Find the intersection point of two lines in (x1,y1,x2,y2) format."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Line equations: ax + by + c = 0
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = x2*y1 - x1*y2
    
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = x4*y3 - x3*y4
    
    det = a1*b2 - a2*b1
    if abs(det) < 1e-8:  # Lines are parallel
        return None
    
    x = (b1*c2 - b2*c1) / det
    y = (a2*c1 - a1*c2) / det
    return (x, y)

def find_dominant_lines(lines, labels, group_idx):
    """Find the most dominant lines in a group based on length."""
    group_lines = [lines[i] for i in range(len(lines)) if labels[i] == group_idx]
    if not group_lines:
        return []
    
    # Calculate line lengths
    lengths = []
    for line in group_lines:
        x1, y1, x2, y2 = line
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        lengths.append(length)
    
    # Sort lines by length (descending)
    sorted_indices = np.argsort(lengths)[::-1]
    
    # Return the top lines (at most 10)
    num_lines = min(10, len(group_lines))
    return [group_lines[i] for i in sorted_indices[:num_lines]]

def find_vanishing_points(lines, labels):
    """Find vanishing points for two groups of lines with improved robustness."""
    if len(lines) < 4:
        return None, None

    # Get most dominant lines from each group for more stable vanishing points
    group1 = find_dominant_lines(lines, labels, 0)
    group2 = find_dominant_lines(lines, labels, 1)
    
    if len(group1) < 2 or len(group2) < 2:
        return None, None
    
    # Find all intersections for each group
    vp1_candidates = []
    for i in range(len(group1)):
        for j in range(i+1, len(group1)):
            vp = find_intersection(group1[i], group1[j])
            if vp is not None:
                vp1_candidates.append(vp)
    
    vp2_candidates = []
    for i in range(len(group2)):
        for j in range(i+1, len(group2)):
            vp = find_intersection(group2[i], group2[j])
            if vp is not None:
                vp2_candidates.append(vp)
    
    # Get median intersection point (more robust than mean)
    if vp1_candidates:
        # Remove outliers before computing median
        vp1_x_values = [p[0] for p in vp1_candidates]
        vp1_y_values = [p[1] for p in vp1_candidates]
        
        # Use IQR to filter outliers
        q1_x, q3_x = np.percentile(vp1_x_values, [25, 75])
        q1_y, q3_y = np.percentile(vp1_y_values, [25, 75])
        iqr_x = q3_x - q1_x
        iqr_y = q3_y - q1_y
        
        filtered_vp1 = [p for p in vp1_candidates 
                        if (q1_x - 1.5*iqr_x <= p[0] <= q3_x + 1.5*iqr_x) and 
                           (q1_y - 1.5*iqr_y <= p[1] <= q3_y + 1.5*iqr_y)]
        
        if filtered_vp1:
            vp1_x = np.median([p[0] for p in filtered_vp1])
            vp1_y = np.median([p[1] for p in filtered_vp1])
            vp1 = (vp1_x, vp1_y)
        else:
            vp1 = None
    else:
        vp1 = None
        
    if vp2_candidates:
        # Remove outliers before computing median
        vp2_x_values = [p[0] for p in vp2_candidates]
        vp2_y_values = [p[1] for p in vp2_candidates]
        
        q1_x, q3_x = np.percentile(vp2_x_values, [25, 75])
        q1_y, q3_y = np.percentile(vp2_y_values, [25, 75])
        iqr_x = q3_x - q1_x
        iqr_y = q3_y - q1_y
        
        filtered_vp2 = [p for p in vp2_candidates 
                        if (q1_x - 1.5*iqr_x <= p[0] <= q3_x + 1.5*iqr_x) and 
                           (q1_y - 1.5*iqr_y <= p[1] <= q3_y + 1.5*iqr_y)]
        
        if filtered_vp2:
            vp2_x = np.median([p[0] for p in filtered_vp2])
            vp2_y = np.median([p[1] for p in filtered_vp2])
            vp2 = (vp2_x, vp2_y)
        else:
            vp2 = None
    else:
        vp2 = None
    
    return vp1, vp2

def find_line_roi(lines, frame_shape):
    """Find region of interest where most lines are concentrated."""
    if not lines:
        h, w = frame_shape[:2]
        return 0, h, 0, w
    
    # Extract all line endpoints
    points = []
    for x1, y1, x2, y2 in lines:
        points.append((x1, y1))
        points.append((x2, y2))
    
    if not points:
        h, w = frame_shape[:2]
        return 0, h, 0, w
    
    # Find the bounding box of these points with some margin
    points = np.array(points)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    
    # Add margin (10% of frame size)
    h, w = frame_shape[:2]
    margin_x = w * 0.1
    margin_y = h * 0.1
    
    roi_left = max(0, min_x - margin_x)
    roi_right = min(w, max_x + margin_x)
    roi_top = max(0, min_y - margin_y)
    roi_bottom = min(h, max_y + margin_y)
    
    return roi_top, roi_bottom, roi_left, roi_right

def draw_grid_from_vanishing_points(frame, vp1, vp2, num_lines=10, tile_size=80):
    """Draw perspective grid and determine the optimal quadrilateral."""
    h, w = frame.shape[:2]
    out = frame.copy()
    
    # Find ROI based on frame size
    roi_top, roi_bottom, roi_left, roi_right = 0, h, 0, w
    
    # Initialize lines and points collections
    horiz_pts = []
    vert_pts = []
    
    # 1) horizontal-like lines from vp1
    for i in range(1, num_lines):
        y = roi_bottom - i * tile_size
        if y < roi_top: break
        if vp1 is None: continue
        
        dx = vp1[0] - 0
        dy = vp1[1] - y
        
        if abs(dy) < 1e-6: continue
        
        # Extend lines far beyond frame boundaries for better intersection estimation
        t_left = (roi_left - 0) / dx if abs(dx) > 1e-6 else float('inf')
        t_right = (roi_right - 0) / dx if abs(dx) > 1e-6 else float('inf')
        
        points = []
        for t in [t_left, t_right]:
            if abs(t) != float('inf'):
                x = 0 + dx * t
                y_t = y + dy * t
                if -3*w <= x <= 4*w and -3*h <= y_t <= 4*h:  # Allow points far outside
                    points.append((int(x), int(y_t)))
        
        if len(points) >= 2:
            cv2.line(out, points[0], points[1], (0,255,0), 1)
            horiz_pts.append((points[0], points[1]))

    # 2) vertical-like lines from vp2
    for i in range(1, num_lines):
        x = i * tile_size
        if x > roi_right: break
        if vp2 is None: continue
        
        dx = vp2[0] - x
        dy = vp2[1] - 0
        
        if abs(dx) < 1e-6: continue
        
        t_top = (roi_top - 0) / dy if abs(dy) > 1e-6 else float('inf')
        t_bottom = (roi_bottom - 0) / dy if abs(dy) > 1e-6 else float('inf')
        
        points = []
        for t in [t_top, t_bottom]:
            if abs(t) != float('inf'):
                x_t = x + dx * t
                y = 0 + dy * t
                if -3*w <= x_t <= 4*w and -3*h <= y <= 4*h:  # Allow points far outside
                    points.append((int(x_t), int(y)))
        
        if len(points) >= 2:
            cv2.line(out, points[0], points[1], (255,0,0), 1)
            vert_pts.append((points[0], points[1]))

    # Draw vanishing points if they're within frame
    if vp1 and -w <= vp1[0] <= 2*w and -h <= vp1[1] <= 2*h:
        vp1_vis = (int(vp1[0]), int(vp1[1]))
        if 0 <= vp1_vis[0] < w and 0 <= vp1_vis[1] < h:
            cv2.circle(out, vp1_vis, 5, (0,255,255), -1)
    
    if vp2 and -w <= vp2[0] <= 2*w and -h <= vp2[1] <= 2*h:
        vp2_vis = (int(vp2[0]), int(vp2[1]))
        if 0 <= vp2_vis[0] < w and 0 <= vp2_vis[1] < h:
            cv2.circle(out, vp2_vis, 5, (255,255,0), -1)

    # Calculate grid corners from line intersections
    grid_corners = None
    if horiz_pts and vert_pts and len(horiz_pts) >= 2 and len(vert_pts) >= 2:
        # Find the best lines to form corners
        horiz_top = horiz_pts[-1]
        horiz_bottom = horiz_pts[0]
        vert_left = vert_pts[0]
        vert_right = vert_pts[-1]
        
        # Convert to line format for intersection function
        h_top = [horiz_top[0][0], horiz_top[0][1], horiz_top[1][0], horiz_top[1][1]]
        h_bottom = [horiz_bottom[0][0], horiz_bottom[0][1], horiz_bottom[1][0], horiz_bottom[1][1]]
        v_left = [vert_left[0][0], vert_left[0][1], vert_left[1][0], vert_left[1][1]]
        v_right = [vert_right[0][0], vert_right[0][1], vert_right[1][0], vert_right[1][1]]
        
        # Calculate the four corners
        bl = find_intersection(h_bottom, v_left)
        br = find_intersection(h_bottom, v_right)
        tr = find_intersection(h_top, v_right)
        tl = find_intersection(h_top, v_left)
        
        # Validate corners more permissively
        if all(p is not None for p in [bl, br, tr, tl]):
            # Allow corners to be far outside the frame but not absurdly far
            valid_corners = True
            for p in [bl, br, tr, tl]:
                if not (-10*w <= p[0] <= 10*w and -10*h <= p[1] <= 10*h):
                    valid_corners = False
                    break
            
            # Check for reasonable quadrilateral
            if valid_corners:
                # Calculate side lengths to check for reasonable proportions
                sides = [
                    np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[1])**2),  # bottom
                    np.sqrt((tr[0]-br[0])**2 + (tr[1]-br[1])**2),  # right
                    np.sqrt((tl[0]-tr[0])**2 + (tl[1]-tr[1])**2),  # top
                    np.sqrt((bl[0]-tl[0])**2 + (bl[1]-tl[1])**2)   # left
                ]
                
                # Sides shouldn't be too imbalanced (no more than 10:1 ratio)
                min_side = min(sides)
                max_side = max(sides)
                
                if min_side > 0 and max_side / min_side < 10:
                    grid_corners = np.array([bl, br, tr, tl], dtype=np.float32)
                    
                    # Draw the detected grid quadrilateral
                    for i in range(4):
                        p1 = (int(grid_corners[i][0]), int(grid_corners[i][1]))
                        p2 = (int(grid_corners[(i+1)%4][0]), int(grid_corners[(i+1)%4][1]))
                        
                        # Only draw if at least one point is inside the frame
                        if ((0 <= p1[0] < w and 0 <= p1[1] < h) or 
                            (0 <= p2[0] < w and 0 <= p2[1] < h)):
                            cv2.line(out, p1, p2, (255, 0, 255), 2)

    # Initialize default corners (full frame)
    default_corners = np.array([
        [0, h],       # bottom-left
        [w, h],       # bottom-right
        [w, 0],       # top-right
        [0, 0]        # top-left
    ], dtype=np.float32)
    
    # Use grid corners if valid, otherwise use full frame
    corners = grid_corners if grid_corners is not None else default_corners
    
    return out, corners, grid_corners is not None

def compute_full_bev(frame, src_pts, output_size=None, use_adaptive_size=True, rectify_whole_image=True):
    """
    Warp the frame according to the floor-plane src_pts → top-down rectangle.
    With improved scaling and proportion handling.
    When rectify_whole_image is True, the entire image is rectified based on the grid transform.
    """
    if src_pts is None:
        return None

    h_src, w_src = frame.shape[:2]
    
    # Calculate the output size based on the source quadrilateral
    if use_adaptive_size:
        # Calculate width and height of the rectified quadrilateral
        # We use the average of the top/bottom and left/right sides
        width1 = np.sqrt((src_pts[1][0] - src_pts[0][0])**2 + (src_pts[1][1] - src_pts[0][1])**2)
        width2 = np.sqrt((src_pts[2][0] - src_pts[3][0])**2 + (src_pts[2][1] - src_pts[3][1])**2)
        avg_width = (width1 + width2) / 2
        
        height1 = np.sqrt((src_pts[3][0] - src_pts[0][0])**2 + (src_pts[3][1] - src_pts[0][1])**2)
        height2 = np.sqrt((src_pts[2][0] - src_pts[1][0])**2 + (src_pts[2][1] - src_pts[1][1])**2)
        avg_height = (height1 + height2) / 2
        
        # Keep aspect ratio proportional to the source shape
        aspect_ratio = avg_width / max(avg_height, 1)
        
        # Adjust output size for whole-image rectification
        if rectify_whole_image:
            # For whole image, maintain source aspect ratio but scale based on detected grid
            scale_factor = max(avg_width, avg_height) / max(w_src, h_src)
            scale_factor = min(2.0, max(0.5, scale_factor))  # Limit scaling between 0.5x and 2x
            output_w = int(w_src * scale_factor)
            output_h = int(h_src * scale_factor)
        else:
            # For just the grid area, use the calculated dimensions
            max_dimension = max(w_src, h_src)
            output_w = int(min(max_dimension, avg_width))
            output_h = int(min(max_dimension, output_w / aspect_ratio))
        
        # Ensure minimum size
        output_w = max(100, output_w)
        output_h = max(100, output_h)
    elif output_size is not None:
        output_w, output_h = output_size
    else:
        output_w, output_h = w_src, h_src

    # For whole image rectification, we need to modify how we handle the transformation
    if rectify_whole_image:
        # Calculate the homography matrix from the grid
        grid_dst_pts = np.array([
            [0.25*output_w, 0.75*output_h],  # bottom-left of grid
            [0.75*output_w, 0.75*output_h],  # bottom-right of grid
            [0.75*output_w, 0.25*output_h],  # top-right of grid
            [0.25*output_w, 0.25*output_h],  # top-left of grid
        ], dtype=np.float32)
        
        # Calculate grid-based homography
        H_grid = cv2.getPerspectiveTransform(src_pts, grid_dst_pts)
        
        # Apply the perspective transformation to the entire image
        bev = cv2.warpPerspective(frame, H_grid, (output_w, output_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0,0,0))
        
        # Draw the grid in the output for reference
        grid_rect = np.array([
            [int(0.25*output_w), int(0.75*output_h)],
            [int(0.75*output_w), int(0.75*output_h)],
            [int(0.75*output_w), int(0.25*output_h)],
            [int(0.25*output_w), int(0.25*output_h)]
        ], dtype=np.int32)
        
        # Draw a thin rectangle on the BEV to show the reference grid area
        for i in range(4):
            cv2.line(bev, tuple(grid_rect[i]), tuple(grid_rect[(i+1)%4]), (0, 255, 0), 1)
    else:
        # Original approach: only rectify the grid area
        dst_pts = np.array([
            [0,           output_h],  # bottom-left
            [output_w,    output_h],  # bottom-right
            [output_w,    0       ],  # top-right
            [0,           0       ],  # top-left
        ], dtype=np.float32)

        # Calculate the homography matrix
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply the perspective transformation
        bev = cv2.warpPerspective(frame, H, (output_w, output_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0,0,0))
    
    return bev

def cluster_lines_by_orientation(lines, frame_shape):
    """Cluster lines into horizontal and vertical groups using custom seed labels."""
    if len(lines) < 2:
        return np.zeros(len(lines), dtype=np.int32)

    # 1) Build the 2-D feature vector [cos(2θ), sin(2θ)] for each line
    features = []
    for x1, y1, x2, y2 in lines:
        theta = np.arctan2((y2 - y1), (x2 - x1))
        features.append([np.cos(2*theta), np.sin(2*theta)])
    features = np.array(features, dtype=np.float32)  # shape = (N, 2)

    # 2) Find “clear” horizontal and vertical lines to seed our centroids
    horizontal_indices = []
    vertical_indices   = []
    for i, (x1, y1, x2, y2) in enumerate(lines):
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx > 3*dy:
            horizontal_indices.append(i)
        if dy > 3*dx:
            vertical_indices.append(i)

    # If we have at least 3 of each, compute their mean-feature as seeds
    init_centers = None
    if len(horizontal_indices) >= 3 and len(vertical_indices) >= 3:
        horiz_feat = features[horizontal_indices].mean(axis=0)
        vert_feat  = features[vertical_indices].mean(axis=0)
        init_centers = np.vstack([horiz_feat, vert_feat])  # shape = (2,2)

    # 3) Prepare the k-means parameters
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # 4) If we have custom centroids, turn them into a per-point init_label array
    if init_centers is not None:
        # compute distances from each feature to each seed
        # dists.shape = (N, 2)
        dists = np.linalg.norm(features[:, None, :] - init_centers[None, :, :], axis=2)
        # assign each point to the nearest seed → init_labels.shape = (N,1)
        init_labels = np.argmin(dists, axis=1).astype(np.int32).reshape(-1, 1)

        # run k-means with your labels
        _, labels, centers = cv2.kmeans(
            features,
            2,
            init_labels,
            crit,
            1,                                # just one iteration is fine
            cv2.KMEANS_USE_INITIAL_LABELS
        )
    else:
        # fallback: let OpenCV pick good centers
        _, labels, centers = cv2.kmeans(
            features,
            2,
            None,
            crit,
            10,
            cv2.KMEANS_PP_CENTERS
        )

    labels = labels.flatten()  # shape = (N,)

    # 5) Ensure cluster 0 is “horizontal-like” and 1 is “vertical-like”
    c1, c2 = centers
    ang1 = 0.5 * np.arctan2(c1[1], c1[0])
    ang2 = 0.5 * np.arctan2(c2[1], c2[0])
    # convert to absolute degrees mod 90
    h1 = abs(np.degrees(ang1)) % 90
    h2 = abs(np.degrees(ang2)) % 90
    if h1 > h2:
        labels = 1 - labels

    return labels

def fit_perspective_grid(frame, lines):
    if len(lines) < 4:
        return frame, None, False

    # Use improved clustering instead of the existing kmeans code
    labels = cluster_lines_by_orientation(lines, frame.shape)
    
    # Create visualization of clustered lines
    clustered_vis = frame.copy()
    for i, (x1, y1, x2, y2) in enumerate(lines):
        # Group 0: Green (horizontal-like), Group 1: Blue (vertical-like)
        color = (0, 255, 0) if labels[i] == 0 else (255, 0, 0)
        cv2.line(clustered_vis, (x1, y1), (x2, y2), color, 2)
    
    # Find vanishing points
    raw_vp1, raw_vp2 = find_vanishing_points(lines, labels)
    vp1, vp2 = get_consistent_vps(raw_vp1, raw_vp2, frame.shape)

    # Draw grid and get corners
    out, corners, is_grid_valid = draw_grid_from_vanishing_points(
        clustered_vis, vp1, vp2, num_lines=20, tile_size=80
    )
    
    return out, corners, is_grid_valid

def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("Error: cannot open video.")
        return

    # Create windows
    cv2.namedWindow("Perspective Grid", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Perspective Grid", 800, 600)
    
    cv2.namedWindow("BEV (Bird's‑Eye View)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BEV (Bird's‑Eye View)", 800, 600)
    
    cv2.namedWindow("BEV Grid Only", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BEV Grid Only", 400, 400)
    
    cv2.namedWindow("Line Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Line Detection", 800, 600)
    
    # Create trackbars for Hough parameters
    cv2.createTrackbar("Canny Low", "Line Detection", 80, 255, lambda x: None)
    cv2.createTrackbar("Canny High", "Line Detection", 150, 255, lambda x: None)
    cv2.createTrackbar("Hough Threshold", "Line Detection", 100, 200, lambda x: None)
    cv2.createTrackbar("Min Line Length", "Line Detection", 160, 200, lambda x: None)
    cv2.createTrackbar("Max Line Gap", "Line Detection", 30, 100, lambda x: None)
    cv2.createTrackbar("Rectify Whole Image", "Perspective Grid", 1, 1, lambda x: None)

    pause = False
    while True:
        k = cv2.waitKey(5) & 0xFF
        if k == ord('q'): break
        if k == ord('p'): pause = not pause
        if pause: continue

        # Get user preference for whole image rectification
        rectify_whole = cv2.getTrackbarPos("Rectify Whole Image", "Perspective Grid") == 1
        
        # Get Hough parameters from trackbars
        canny_low = cv2.getTrackbarPos("Canny Low", "Line Detection")
        canny_high = cv2.getTrackbarPos("Canny High", "Line Detection")
        hough_threshold = cv2.getTrackbarPos("Hough Threshold", "Line Detection")
        min_line_length = cv2.getTrackbarPos("Min Line Length", "Line Detection")
        max_line_gap = cv2.getTrackbarPos("Max Line Gap", "Line Detection")

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Detect lines and get visualization
        raw_lines, lines_vis, edges, debug_vis = detect_hough_lines(
            frame, 
            canny_low=canny_low,
            canny_high=canny_high,
            hough_threshold=hough_threshold,
            min_line_length=min_line_length,
            max_line_gap=max_line_gap
        )
        
        # Show line detection visualization
        cv2.imshow("Line Detection", debug_vis)

        # Fit perspective grid
        grid_vis, corners, is_grid_valid = fit_perspective_grid(frame, raw_lines)

        # Show grid visualization
        cv2.imshow("Perspective Grid", grid_vis)

        if corners is not None:
            # Show both rectification types
            # Full image rectification
            whole_bev = compute_full_bev(frame, corners, output_size=None, 
                                        use_adaptive_size=True, 
                                        rectify_whole_image=True)
            if whole_bev is not None:
                cv2.imshow("BEV (Bird's‑Eye View)", whole_bev)
            
            # Grid-only rectification (for comparison)
            grid_only_bev = compute_full_bev(frame, corners, output_size=None, 
                                           use_adaptive_size=True, 
                                           rectify_whole_image=False)
            if grid_only_bev is not None:
                cv2.imshow("BEV Grid Only", grid_only_bev)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
