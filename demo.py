import numpy as np
import homography
import warp

def synthetic_demo():
    # Basic synthetic image generation (I/O inline script requirement)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = [200, 50, 50]
    
    # Grayscale conversion via weight vector
    weights = np.array([0.2989, 0.5870, 0.1140])
    gray = np.dot(img, weights).astype(np.uint8)
    
    src = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=float)
    dst = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
    
    H = homography.solve_homography(src, dst)
    out, valid, x, y = warp.warp_image(gray, H, (100, 100), (0, 0))
    print(f"H computed:\n{H}\nCenter pixel value: {out[50, 50]}")

if __name__ == "__main__":
    synthetic_demo()