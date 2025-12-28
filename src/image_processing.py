import cv2
import numpy as np
import os

# Load the template once when the module imports to improve performance
TEMPLATE_PATH = os.path.join("images", "zero_template.jpg")
ZERO_TEMPLATE = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)

# Fallback: If file is missing or path is wrong, create a black image to prevent crash
if ZERO_TEMPLATE is None:
    print(f"Warning: Could not load {TEMPLATE_PATH}. Using generated black image.")
    ZERO_TEMPLATE = np.zeros((81, 81), dtype=np.uint8)

def crop_center(binary_img, gray_img, cropx, cropy):
    """
    Logic from your code lines 122-143.
    Uses 'images/zero_template.jpg' for empty cells.
    """
    y, x = binary_img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)

    crop_bin = binary_img[starty:starty+cropy, startx:startx+cropx]
    crop_gry = gray_img[starty:starty+cropy, startx:startx+cropx]

    contours, _ = cv2.findContours(crop_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Use the globally loaded template
    # Resize it just in case the template on disk isn't exactly cropx by cropy
    template_resized = cv2.resize(ZERO_TEMPLATE, (cropx, cropy))

    if len(contours) == 0:
        return template_resized

    max_cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_cnt) < 250:
        return template_resized

    x, y, w, h = cv2.boundingRect(max_cnt)
    d = (h - w) // 2
    c_h, c_w = crop_gry.shape
    
    # Safe slicing with bounds
    y_start = y
    y_end = y + h
    x_start = max(0, x - d)
    x_end = min(c_w, x + w + d)
    
    roi = crop_gry[y_start:y_end, x_start:x_end]
    return roi

def process_pipeline(image_bytes):
    """
    Combines your preprocessing steps (Lines 60-160).
    Returns: Array of shape (81, 32, 32, 1) normalized 0-1.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Resize to specific size used in your code
    img = cv2.resize(img, (1026, 1026))
    
    # Grayscale & Blur
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgray = cv2.GaussianBlur(imgray, (11, 11), 0)
    
    # Threshold
    thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 5, 2)
    
    # Dilate
    kernel = np.array([0,1,0,1,1,1,0,1,0], dtype=np.uint8).reshape(3,3)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find Grid Contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found")
        
    cnt1 = max(contours, key=cv2.contourArea)
    epsilon = 0.1 * cv2.arcLength(cnt1, True)
    cnt = cv2.approxPolyDP(cnt1, epsilon, True).squeeze()
    
    # Sort corners (Top-Left, Top-Right, Bot-Right, Bot-Left)
    cor_list = sorted(cnt.tolist(), key=lambda c: c[1])
    ord_list = sorted(cor_list[:2], key=lambda c: c[0]) + sorted(cor_list[2:], key=lambda c: c[0])
    
    # Perspective Transform
    pts1 = np.float32(ord_list)
    pts2 = np.float32([[0,0], [1026,0], [0,1026], [1026,1026]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    out_gry = cv2.warpPerspective(img, M, (1026, 1026))
    out_bin = cv2.warpPerspective(thresh, M, (1026, 1026))
    
    # Dilate again
    out_bin = cv2.dilate(out_bin, kernel, iterations=1)
    out_gry_gray = cv2.cvtColor(out_gry, cv2.COLOR_BGR2GRAY)

    # Split into 81 blocks
    H, W = 1026 // 9, 1026 // 9
    sudokus = []
    
    for y in range(0, 1026, H):
        for x in range(0, 1026, W):
            tiles_bin = out_bin[y:y+H, x:x+W]
            tiles_gry = out_gry_gray[y:y+H, x:x+W]
            
            # Crop digit from block
            digit = crop_center(tiles_bin, tiles_gry, 81, 81)
            
            # Resize to 32x32 for Model
            digit = cv2.resize(digit, (32, 32), interpolation=cv2.INTER_AREA)
            sudokus.append(digit / 255.0) # Normalize
            
    # Reshape for model (81, 32, 32, 1)
    sudoku_numbers = np.array(sudokus, dtype=np.float32).reshape(81, 32, 32, 1)
    return sudoku_numbers