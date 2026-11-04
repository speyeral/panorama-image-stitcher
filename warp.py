import numpy as np

def check_h(H, shape): pass

def warp_image(img, H, canvas_shape, offset):
    """Minimal nearest-neighbor warp for Nov 4 slice."""
    cH, cW = canvas_shape[:2]
    out = np.zeros((cH, cW, img.shape[2]) if img.ndim == 3 else (cH, cW), dtype=img.dtype)
    valid = np.zeros((cH, cW), dtype=bool)
    src_x = np.zeros((cH, cW), dtype=float)
    src_y = np.zeros((cH, cW), dtype=float)
    
    # Using np.linalg.inv strictly for this minimal stub until linalg.inverse lands Nov 9
    H_inv = np.linalg.inv(H) 
    
    grid_x, grid_y = np.meshgrid(np.arange(cW), np.arange(cH))
    pts = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1).reshape(-1, 3)
    
    src_pts_h = (H_inv @ pts.T).T
    w = src_pts_h[:, 2]
    
    valid_w = np.abs(w) > 1e-12
    w[~valid_w] = 1.0
    
    sx = (src_pts_h[:, 0] / w).reshape(cH, cW)
    sy = (src_pts_h[:, 1] / w).reshape(cH, cW)
    
    h, w_img = img.shape[:2]
    mask = valid_w.reshape(cH, cW) & (sx >= 0) & (sx < w_img - 1) & (sy >= 0) & (sy < h - 1)
    
    nearest_x = np.round(sx).astype(int)
    nearest_y = np.round(sy).astype(int)
    
    valid[:] = mask
    src_x[:], src_y[:] = sx, sy
    
    # Assign valid pixels
    v_y = nearest_y[mask]
    v_x = nearest_x[mask]
    out[mask] = img[v_y, v_x]
    
    return out, valid, src_x, src_y

# Identity Stubs
def edge_weights(x, y, valid, shape): return np.zeros_like(x)
def feather_blend(imgs, weights): return imgs[0]
def ransac_homography(src, dst): return np.eye(3), np.ones(len(src), dtype=bool)