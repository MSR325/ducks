import sys
import cv2
import numpy as np

from rect import *
from stabilizer import HomographyStabilizer

import os
from tqdm import tqdm
import collections

FROM = 0
VIDEO = 'patos.mp4'
OUTPUT_DIR = 'rectified_frames'
HISTORY_SIZE = 5


def is_vp_valid(vp, frame_shape, max_factor=10):
    h, w = frame_shape[:2]
    max_x, max_y = w * max_factor, h * max_factor
    min_x, min_y = -max_x, -max_y
    
    return (min_x <= vp[0] <= max_x and 
            min_y <= vp[1] <= max_y and
            not np.isinf(vp[0]) and not np.isinf(vp[1]) and
            not np.isnan(vp[0]) and not np.isnan(vp[1]))

def select_consistent_vp(vp1, vp2, history, frame_shape):
    if not history:
        angles1 = np.arctan2(np.abs(vp1[1]), np.abs(vp1[0]))
        angles2 = np.arctan2(np.abs(vp2[1]), np.abs(vp2[0]))
        
        if angles1 < angles2:
            return vp1, vp2
        else:
            return vp2, vp1
    
    hist_vpx = [entry['vpx'] for entry in history]
    hist_vpy = [entry['vpy'] for entry in history]
    
    score_1x = sum(np.sqrt((vp1[0] - h[0])**2 + (vp1[1] - h[1])**2) 
                  for h in hist_vpx if is_vp_valid(h, frame_shape))
    score_1y = sum(np.sqrt((vp1[0] - h[0])**2 + (vp1[1] - h[1])**2) 
                  for h in hist_vpy if is_vp_valid(h, frame_shape))
    score_2x = sum(np.sqrt((vp2[0] - h[0])**2 + (vp2[1] - h[1])**2) 
                  for h in hist_vpx if is_vp_valid(h, frame_shape))
    score_2y = sum(np.sqrt((vp2[0] - h[0])**2 + (vp2[1] - h[1])**2) 
                  for h in hist_vpy if is_vp_valid(h, frame_shape))
    
    if (score_1x + score_2y) < (score_2x + score_1y):
        return vp1, vp2
    else:
        return vp2, vp1

def calculate_homography_similarity(H1, H2):
    """Calculate similarity between two homography matrices."""
    if H1 is None or H2 is None:
        return 0
    
    # Normalize matrices for better comparison
    H1_norm = H1 / np.linalg.norm(H1)
    H2_norm = H2 / np.linalg.norm(H2)
    
    # Compute similarity score (higher is better)
    return np.abs(np.sum(H1_norm * H2_norm))

def calculate_vp_angle(vp, frame_shape):
    center_x, center_y = frame_shape[1] // 2, frame_shape[0] // 2
    return np.arctan2(vp[1] - center_y, vp[0] - center_x)

def improved_metric_rectification(image, vpx, vpy, prev_homography=None, clip=True, clip_factor=3):
    """
    Improved metric rectification that directly enforces square aspect ratio.
    This implementation ensures that squares in the scene remain squares after rectification.
    """
    # Step 1: Initial projective rectification
    vanishing_line = np.cross(vpx, vpy)
    H_proj = np.eye(3)
    H_proj[2] = vanishing_line / vanishing_line[2]
    H_proj = H_proj / H_proj[2, 2]

    # Critical step: Check if vanishing points are orthogonal in 3D
    # If they are, they should be orthogonal after removal of projective distortion
    # assuming they represent orthogonal directions in the world
    
    # Calculate the circular points (ideal points that encode metric structure)
    # For orthogonal vanishing points under perspective, we know their cross-ratio
    # We'll use this to recover the affine transformation that enforces orthogonality
    
    # First, get the transformed vanishing points
    vp1_proj = np.dot(H_proj, vpx)
    vp2_proj = np.dot(H_proj, vpy)
    
    # Normalize them
    vp1_proj = vp1_proj / vp1_proj[2]
    vp2_proj = vp2_proj / vp2_proj[2]
    
    # Calculate vectors from origin to vanishing points
    v1 = vp1_proj[:2]
    v2 = vp2_proj[:2]
    
    # These should be orthogonal if the scene has orthogonal directions
    # We'll create an affine transformation that enforces this
    
    # DIRECT APPROACH: Let's use the SVD to find the closest orthogonal basis
    # Stack the vectors as columns
    M = np.column_stack([v1, v2])
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    # Correct aspect ratio by forcing singular values to be equal
    S_corrected = np.array([np.sqrt(S[0] * S[1]), np.sqrt(S[0] * S[1])])
    
    # Reconstruct with corrected singular values
    M_corrected = U @ np.diag(S_corrected) @ Vt
    
    # Get the corrected directions
    v1_corrected = M_corrected[:, 0]
    v2_corrected = M_corrected[:, 1]
    
    # Check orthogonality (should be close to zero)
    dot_product = np.dot(v1_corrected, v2_corrected)
    
    # Now create an affine transformation that maps v1, v2 to v1_corrected, v2_corrected
    A_metric = np.eye(3)
    A_metric[:2, :2] = np.column_stack([v1_corrected, v2_corrected]) @ np.linalg.inv(np.column_stack([v1, v2]))
    
    # Combine with projective homography
    H_combined = np.dot(A_metric, H_proj)
    
    # Multiple configuration approach - generate several candidates
    # We'll create variations by rotating the basis slightly
    candidates = []
    
    # Create the base candidate
    try:
        # Compute translation to ensure all of image is visible
        cords = np.dot(H_combined, [[0, 0, image.shape[1], image.shape[1]],
                                   [0, image.shape[0], 0, image.shape[0]],
                                   [1, 1, 1, 1]])
        cords = cords[:2] / cords[2]
        
        tx = min(0, cords[0].min())
        ty = min(0, cords[1].min())
        
        # Apply clipping if requested
        if clip:
            max_offset = max(image.shape) * clip_factor / 2
            tx = max(tx, -max_offset)
            ty = max(ty, -max_offset)
        
        T = np.array([[1, 0, -tx],
                      [0, 1, -ty],
                      [0, 0, 1]])
        
        base_H = np.dot(T, H_combined)
        
        # Calculate dimensions of output image
        warp_cords = np.dot(base_H, [[0, 0, image.shape[1], image.shape[1]],
                                     [0, image.shape[0], 0, image.shape[0]],
                                     [1, 1, 1, 1]])
        warp_cords = warp_cords[:2] / warp_cords[2]
        max_x = int(warp_cords[0].max() - warp_cords[0].min())
        max_y = int(warp_cords[1].max() - warp_cords[1].min())
        
        candidates.append({
            'homography': base_H,
            'dimensions': (max_y, max_x),
            'ortho_score': abs(dot_product),  # Lower is better
            'scale_ratio': S_corrected[1] / S_corrected[0]  # Closer to 1 is better
        })
        
        # Create variations by applying small rotations
        for angle in [-5, -2.5, 2.5, 5]:  # Angles in degrees
            theta = np.radians(angle)
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            
            # Apply rotation to base homography
            var_H = np.dot(rot_matrix, base_H)
            
            # Calculate dimensions
            warp_cords = np.dot(var_H, [[0, 0, image.shape[1], image.shape[1]],
                                       [0, image.shape[0], 0, image.shape[0]],
                                       [1, 1, 1, 1]])
            warp_cords = warp_cords[:2] / warp_cords[2]
            max_x = int(warp_cords[0].max() - warp_cords[0].min())
            max_y = int(warp_cords[1].max() - warp_cords[1].min())
            
            candidates.append({
                'homography': var_H,
                'dimensions': (max_y, max_x),
                'ortho_score': abs(dot_product),  # All variations have same ortho score
                'scale_ratio': S_corrected[1] / S_corrected[0]  # All variations have same scale ratio
            })
            
        # Also try flipping the axes if needed
        flipped_A = A_metric.copy()
        flipped_A[:2, 0] = A_metric[:2, 1]
        flipped_A[:2, 1] = A_metric[:2, 0]
        
        flipped_H = np.dot(flipped_A, H_proj)
        
        # Apply translation
        cords = np.dot(flipped_H, [[0, 0, image.shape[1], image.shape[1]],
                                  [0, image.shape[0], 0, image.shape[0]],
                                  [1, 1, 1, 1]])
        cords = cords[:2] / cords[2]
        
        tx = min(0, cords[0].min())
        ty = min(0, cords[1].min())
        
        # Apply clipping if requested
        if clip:
            max_offset = max(image.shape) * clip_factor / 2
            tx = max(tx, -max_offset)
            ty = max(ty, -max_offset)
        
        T = np.array([[1, 0, -tx],
                      [0, 1, -ty],
                      [0, 0, 1]])
        
        flipped_H_final = np.dot(T, flipped_H)
        
        # Calculate dimensions
        warp_cords = np.dot(flipped_H_final, [[0, 0, image.shape[1], image.shape[1]],
                                             [0, image.shape[0], 0, image.shape[0]],
                                             [1, 1, 1, 1]])
        warp_cords = warp_cords[:2] / warp_cords[2]
        max_x = int(warp_cords[0].max() - warp_cords[0].min())
        max_y = int(warp_cords[1].max() - warp_cords[1].min())
        
        candidates.append({
            'homography': flipped_H_final,
            'dimensions': (max_y, max_x),
            'ortho_score': abs(dot_product),  # Same ortho score
            'scale_ratio': S_corrected[1] / S_corrected[0]  # Same scale ratio
        })
        
    except Exception as e:
        print(f"Error creating base candidate: {str(e)}")
        # If we can't create the base candidate, try a traditional approach
        return traditional_metric_rectification(image, vpx, vpy, prev_homography, clip, clip_factor)
    
    # If no candidates, use traditional method
    if not candidates:
        return traditional_metric_rectification(image, vpx, vpy, prev_homography, clip, clip_factor)
    
    # If no previous homography, use the best candidate
    if prev_homography is None:
        selected = candidates[0]  # Base candidate is usually best
        final_homography = selected['homography']
        dimensions = selected['dimensions']
    else:
        # With previous homography, balance metric quality with temporal consistency
        best_score = float('inf')
        best_candidate = None
        
        for candidate in candidates:
            # Calculate similarity to previous homography
            H_diff = candidate['homography'] - prev_homography
            temporal_score = np.linalg.norm(H_diff[:2, :2])
            
            # Check for sign flips in critical elements
            sign_flips = 0
            for i in range(2):
                for j in range(2):
                    if (np.sign(candidate['homography'][i, j]) != np.sign(prev_homography[i, j]) and 
                        abs(candidate['homography'][i, j]) > 0.1 and 
                        abs(prev_homography[i, j]) > 0.1):
                        sign_flips += 1
            
            # Heavily penalize sign flips
            combined_score = temporal_score + sign_flips * 10.0
            
            if combined_score < best_score:
                best_score = combined_score
                best_candidate = candidate
        
        final_homography = best_candidate['homography']
        dimensions = best_candidate['dimensions']
    
    # Warp the image
    from skimage import transform
    warped_img = transform.warp(image, np.linalg.inv(final_homography),
                              output_shape=dimensions,
                              order=1)
    
    return warped_img, final_homography


def traditional_metric_rectification(image, vpx, vpy, prev_homography=None, clip=True, clip_factor=3):
    """Fallback method using traditional approach to metric rectification."""
    # Step 1: Initial projective rectification
    vanishing_line = np.cross(vpx, vpy)
    H = np.eye(3)
    H[2] = vanishing_line / vanishing_line[2]
    H = H / H[2, 2]

    v_post1 = np.dot(H, vpx)
    v_post2 = np.dot(H, vpy)
    v_post1 = v_post1 / np.sqrt(v_post1[0]**2 + v_post1[1]**2)
    v_post2 = v_post2 / np.sqrt(v_post2[0]**2 + v_post2[1]**2)

    directions = np.array([[v_post1[0], -v_post1[0], v_post2[0], -v_post2[0]],
                          [v_post1[1], -v_post1[1], v_post2[1], -v_post2[1]]])

    thetas = np.arctan2(directions[0], directions[1])
    
    # Find direction closest to horizontal axis
    h_ind = np.argmin(np.abs(thetas))
    
    # Find positve angle among the rest for the vertical axis
    if h_ind // 2 == 0:
        v_ind = 2 + np.argmax([thetas[2], thetas[3]])
    else:
        v_ind = np.argmax([thetas[0], thetas[1]])
    
    # Create affine transformation enforcing orthogonality
    A1 = np.array([[directions[0, v_ind], directions[0, h_ind], 0],
                   [directions[1, v_ind], directions[1, h_ind], 0],
                   [0, 0, 1]])
    
    # Ensure positive determinant
    if np.linalg.det(A1[:2, :2]) < 0:
        A1[:, 0] = -A1[:, 0]
    
    # Get inverse
    A = np.linalg.inv(A1)
    
    # Get scaling factors
    sx = np.sqrt(A[0, 0]**2 + A[0, 1]**2)
    sy = np.sqrt(A[1, 0]**2 + A[1, 1]**2)
    
    # Force equal scaling (this ensures squares remain squares)
    s_avg = (sx + sy) / 2
    A_metric = A.copy()
    A_metric[0, :2] = A[0, :2] * (s_avg / sx)
    A_metric[1, :2] = A[1, :2] * (s_avg / sy)
    
    # Combine with projective homography
    H_combined = np.dot(A_metric, H)
    
    # Apply translation to ensure all of image is visible
    cords = np.dot(H_combined, [[0, 0, image.shape[1], image.shape[1]],
                               [0, image.shape[0], 0, image.shape[0]],
                               [1, 1, 1, 1]])
    cords = cords[:2] / cords[2]
    
    tx = min(0, cords[0].min())
    ty = min(0, cords[1].min())
    
    # Apply clipping if requested
    if clip:
        max_offset = max(image.shape) * clip_factor / 2
        tx = max(tx, -max_offset)
        ty = max(ty, -max_offset)
    
    T = np.array([[1, 0, -tx],
                  [0, 1, -ty],
                  [0, 0, 1]])
    
    final_homography = np.dot(T, H_combined)
    
    # Determine output image size
    warp_cords = np.dot(final_homography, [[0, 0, image.shape[1], image.shape[1]],
                                       [0, image.shape[0], 0, image.shape[0]],
                                       [1, 1, 1, 1]])
    warp_cords = warp_cords[:2] / warp_cords[2]
    max_x = int(warp_cords[0].max() - warp_cords[0].min())
    max_y = int(warp_cords[1].max() - warp_cords[1].min())
    
    # Warp the image
    from skimage import transform
    warped_img = transform.warp(image, np.linalg.inv(final_homography),
                              output_shape=(max_y, max_x),
                              order=1)
    
    return warped_img, final_homography


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("Error: cannot open video.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, FROM)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    history = collections.deque(maxlen=HISTORY_SIZE)
    prev_homography = None
    last_successful_vpx = None
    last_successful_vpy = None

    progress = tqdm(range(FROM, total_frames), desc="Rectifying frames", total=total_frames, initial=FROM)

    for frame_idx in progress:
        ret, frame = cap.read()
        if not ret:
            break
        
        edgelets = compute_edgelets(frame, sigma=1)
        vp1 = ransac_vanishing_point(edgelets, num_ransac_iter=1000, threshold_inlier=2)
        vp1 = reestimate_model(vp1, edgelets, threshold_reestimate=5)
        
        edgelets2 = remove_inliers(vp1, edgelets, 10)
        vp2 = ransac_vanishing_point(edgelets2, num_ransac_iter=1000, threshold_inlier=2)
        vp2 = reestimate_model(vp2, edgelets2, threshold_reestimate=5)
        
        vp1_valid = is_vp_valid(vp1, frame.shape)
        vp2_valid = is_vp_valid(vp2, frame.shape)
        
        if not vp1_valid or not vp2_valid:
            if last_successful_vpx is not None and last_successful_vpy is not None:
                vpx, vpy = last_successful_vpx, last_successful_vpy
                print(f"Using last successful VPs for frame {frame_idx} due to invalid VPs")
            else:
                print(f"Warning: Invalid VPs detected in frame {frame_idx} and no history available")
                continue
        else:
            vpx, vpy = select_consistent_vp(vp1, vp2, history, frame.shape)
            last_successful_vpx, last_successful_vpy = vpx, vpy
        
        try:
            # Use the improved metric rectification method
            warped_img, current_homography = improved_metric_rectification(
                frame, vpx, vpy, prev_homography, clip=True, clip_factor=1.5)

            progress.refresh()
            
            # Update previous homography
            prev_homography = current_homography
            
            if warped_img.dtype != np.uint8:
                warped_img = (warped_img * 255).astype(np.uint8)
            
            frame_filename = os.path.join(OUTPUT_DIR, f'frame_{frame_idx:06d}.jpg')
            cv2.imwrite(frame_filename, warped_img)
            
            vpx_angle = calculate_vp_angle(vpx, frame.shape)
            vpy_angle = calculate_vp_angle(vpy, frame.shape)
            
            history.append({
                'vpx': vpx,
                'vpy': vpy,
                'vpx_angle': vpx_angle,
                'vpy_angle': vpy_angle,
                'frame_idx': frame_idx
            })
            
            cv2.imshow("warped image", warped_img)
            
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
                
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {str(e)}")
            continue
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Frames saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
