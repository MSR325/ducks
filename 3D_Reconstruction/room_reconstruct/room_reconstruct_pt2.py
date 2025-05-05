import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import trimesh

# Load the image
image_path = "3D_Reconstruction/room_reconstruct/test_imgs/striped.jpg"
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale and detect edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 20, 120, apertureSize=3)

# Detect lines with Hough Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10)

# Image dimensions
height, width = img.shape[:2]
left_wall_x_range = (0, width * 0.5)
right_wall_x_range = (width * 0.5, width)
bottom_exclusion_zone = height * 0.75

# Thresholds
horizontal_thresh = 3
vertical_thresh = 15
oblique_thresh_min = 20
oblique_thresh_max = 70

# Containers for line categories
img_lines = img_rgb.copy()
img_horizontal = img_rgb.copy()
img_vertical = img_rgb.copy()
img_left_wall = img_rgb.copy()
img_right_wall = img_rgb.copy()

left_wall_lines = []
right_wall_lines = []
horizontal_lines = []
vertical_lines = []

def make_double_sided(mesh):
    flipped = o3d.geometry.TriangleMesh()
    flipped.vertices = mesh.vertices
    flipped.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, [0, 2, 1]])  # Flip winding
    mesh += flipped
    return mesh

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        avg_x = (x1 + x2) / 2
        avg_y = (y1 + y2) / 2

        cv2.line(img_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if angle < horizontal_thresh or angle > (180 - horizontal_thresh):
            cv2.line(img_horizontal, (x1, y1), (x2, y2), (0, 255, 0), 2)
            horizontal_lines.append([(x1, y1), (x2, y2)])
        elif abs(angle - 90) < vertical_thresh:
            cv2.line(img_vertical, (x1, y1), (x2, y2), (0, 0, 255), 2)
            vertical_lines.append([(x1, y1), (x2, y2)])

        if (oblique_thresh_min < angle < oblique_thresh_max or
            (180 - oblique_thresh_max) < angle < (180 - oblique_thresh_min)):
            if left_wall_x_range[0] <= avg_x <= left_wall_x_range[1] and avg_y < bottom_exclusion_zone:
                cv2.line(img_left_wall, (x1, y1), (x2, y2), (255, 0, 255), 2)
                left_wall_lines.append([(x1, y1), (x2, y2)])
            elif right_wall_x_range[0] <= avg_x <= right_wall_x_range[1] and avg_y < bottom_exclusion_zone:
                cv2.line(img_right_wall, (x1, y1), (x2, y2), (0, 255, 255), 2)
                right_wall_lines.append([(x1, y1), (x2, y2)])

# Step 1: Vanishing point estimation
def compute_vanishing_point(lines):
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            (x1, y1), (x2, y2) = lines[i]
            (x3, y3), (x4, y4) = lines[j]
            l1 = np.cross([x1, y1, 1], [x2, y2, 1])
            l2 = np.cross([x3, y3, 1], [x4, y4, 1])
            p = np.cross(l1, l2)
            if p[2] != 0:
                intersections.append(p[:2] / p[2])
    return np.median(intersections, axis=0) if intersections else None

vp_left = compute_vanishing_point(left_wall_lines)
vp_right = compute_vanishing_point(right_wall_lines)
vp_front = compute_vanishing_point(horizontal_lines)

# Step 2: Estimate 3D axes (directions)
def direction(vp):
    return vp / np.linalg.norm(vp) if vp is not None else None

dir_x = direction(vp_right - vp_left) if vp_right is not None and vp_left is not None else None
dir_y = direction(vp_front - ((vp_right + vp_left) / 2)) if vp_front is not None and vp_left is not None and vp_right is not None else None

# Step 3: Camera intrinsics
fx = fy = 1000
cx, cy = width / 2, height / 2
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# Step 4: Generate mock 3D wall planes
walls = []
wall_depth = 2.0
wall_height = 1.0  # you can change this
wall_width = 1.0   # width of the front wall

# Front wall (XY plane at z=0)
walls.append(np.array([
    [0, 0, 0],
    [wall_width, 0, 0],
    [wall_width, wall_height, 0],
    [0, wall_height, 0]
]))

# Left wall (YZ plane at x=0)
walls.append(np.array([
    [0, 0, 0],
    [0, wall_height, 0],
    [0, wall_height, -wall_depth],
    [0, 0, -wall_depth]
]))

# Right wall (YZ plane at x=wall_width)
walls.append(np.array([
    [wall_width, 0, 0],
    [wall_width, wall_height, 0],
    [wall_width, wall_height, -wall_depth],
    [wall_width, 0, -wall_depth]
]))

# Floor (XZ plane at y=0)
walls.append(np.array([
    [0, 0, 0],
    [wall_width, 0, 0],
    [wall_width, 0, -wall_depth],
    [0, 0, -wall_depth]
]))


# Step 5: Export as 3D mesh
pcd = o3d.geometry.TriangleMesh()

for wall in walls:
    v = o3d.utility.Vector3dVector(wall)
    faces = [[0, 1, 2], [2, 3, 0]]
    f = o3d.utility.Vector3iVector(faces)
    mesh = o3d.geometry.TriangleMesh(v, f)
    pcd += mesh

pcd.compute_vertex_normals()
pcd.compute_vertex_normals()
pcd = make_double_sided(pcd)

# Save PLY and OBJ
ply_path = "RETO_PATOS/3D_Reconstruction/out/reconstructed_room.ply"
dae_path = "RETO_PATOS/3D_Reconstruction/out/reconstructed_room2.dae"
stl_path = "RETO_PATOS/3D_Reconstruction/out/reconstructed_room2.stl"

# Load mesh from existing PLY
mesh = trimesh.load(ply_path)

# Export to STL
mesh.export(stl_path)

# Export to DAE (requires `pycollada` as backend)
mesh.export(dae_path)

print("✅ Exported meshes")

# Plotting
fig, axs = plt.subplots(2, 4, figsize=(22, 10))
axs[0, 0].imshow(img_rgb); axs[0, 0].set_title("Original")
axs[0, 1].imshow(edges, cmap='gray'); axs[0, 1].set_title("Canny Edges")
axs[0, 2].imshow(img_lines); axs[0, 2].set_title("All Hough Lines")
axs[0, 3].imshow(img_horizontal); axs[0, 3].set_title("Horizontal Lines")
axs[1, 0].imshow(img_vertical); axs[1, 0].set_title("Vertical Lines")
axs[1, 1].imshow(img_left_wall); axs[1, 1].set_title("Left Wall Lines")
axs[1, 2].imshow(img_right_wall); axs[1, 2].set_title("Right Wall Lines")
axs[1, 3].axis("off")
for ax in axs.flatten(): ax.axis("off")
plt.tight_layout(); plt.show()
