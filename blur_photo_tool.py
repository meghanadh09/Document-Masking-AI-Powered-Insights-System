import cv2
import pytesseract
import spacy
import numpy as np
import os
import re

# --- For Windows Users: Update this path if Tesseract is installed elsewhere ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Load Models ---
# Load spaCy's English model
nlp = spacy.load('en_core_web_sm')

try:
    # Download the YuNet model if you don't have it:
    # wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
    face_detector = cv2.FaceDetectorYN.create(
        'face_detection_yunet_2023mar.onnx',
        "",
        (320, 320)
    )
    use_yunet_detector = True
    use_dnn_detector = False
    print("Using YuNet face detector (most accurate)")
except:
    try:
        # Fallback to older DNN model
        face_net = cv2.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')
        use_dnn_detector = True
        use_yunet_detector = False
        print("Using DNN face detector (more accurate)")
    except:
        # Fallback to Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        use_dnn_detector = False
        use_yunet_detector = False
        print("Using Haar Cascade face detector (fallback)")

# --- Configuration ---
# Define which text entities are considered sensitive
SENSITIVE_ENTITIES = ["PERSON", "GPE", "ORG", "DATE", "LOC"]

SENSITIVE_REGEX_PATTERNS = [
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?', # IPv4 Address with optional CIDR (e.g., 192.168.1.1, 0.0.0.0/0)
    r'\d{3}-\d{3}-\d{4}',                 # Phone numbers (e.g., 123-456-7890)
    r'\d{10}',                            # 10-digit numbers (e.g., some phone numbers, IDs)
    r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}', # Credit card like numbers (e.g., 1111-2222-3333-4444)
    r'\b[A-Z0-9]{16}\b',                  # 16-character alphanumeric IDs
    r'\b\d{12}\b',                        # 12-digit numbers (e.g., AWS account IDs in ARNs like 123456789012)
    r'AKIA[0-9A-Z]{16}',                  # AWS Access Key IDs (e.g., AKIAIOSFODNN7EXAMPLE)
    r't-0-[0-9a-f]{17}',                  # AWS session IDs (e.g., t-0-1234567890abcdefg)
    r'arn:aws:iam::\d{12}:',              # AWS ARN with account ID (e.g., arn:aws:iam::123456789012:user/test)
    r'\b(\d{1,2}|[A-Z]{3})\s+\d{1,2},\s+\d{4}\b', # Dates like "May 24, 2023" (if not caught by spaCy)
    r'(?:tcp|udp):\d{1,5}',               # Ports like tcp:22, udp:53, tcp:8080
    r'\b\d{1,5}-\d{1,5}\b',               # Port ranges like 0-65535
    r'10\.128\.0\.0/9',                   # IP CIDR blocks (kept original as example)
    r'"Effect"\s*:\s*"Allow"',            # AWS bucket policies with "Effect":"Allow"
    r'"effect"\s*:\s*"Allow"',            # AWS bucket policies with "effect":"Allow" (lowercase)
    r'(?i)\b(?:room|office|conference|meeting|kitchen|bathroom|restroom|lobby|reception|storage|server|closet|break|cafeteria|training|boardroom|executive|admin|hr|it|finance|marketing|sales|legal|warehouse|lab|laboratory|workshop|studio|lounge|waiting|entrance|exit|stairwell|elevator|hallway|corridor)\s+(?:[A-Z]?\d+[A-Z]?|\d+|\w+)', # Room names/numbers
    r'\b[A-Z]{1,2}\d{1,4}[A-Z]?\b',       # Room codes like A101, B2, C405A
    r'(?i)\b(?:north|south|east|west|main|upper|lower|ground|first|second|third|basement|sub)\s+(?:wing|floor|level|building|block)\b', # Building sections
    r'(?i)\bbuilding\s+[A-Z0-9]+\b',      # Building identifiers like "Building A", "Building 123"
    # Network zone terminology
    r'(?i)\b(?:internet|dmz|trusted|privileged|untrusted)\b', # Network security zones
    r'(?i)\b(?:vpn\s+gateway|load\s+balancer|lb|firewall|proxy)\b', # Network components
    r'(?i)\b(?:internal\s+services|application\s+services|mainframe)\b', # Service types
    r'(?i)\b(?:remote\s+employee|remote\s+office|mobile\s+devices|untrusted\s+client)\b', # Client types
    # Network infrastructure terms
    r'(?i)\b(?:gateway|router|switch|hub|access\s+point|ap)\b', # Network hardware
    r'(?i)\b(?:vlan|subnet|network\s+segment|security\s+zone)\b', # Network segmentation
    r'(?i)\b(?:intrusion\s+detection|ids|intrusion\s+prevention|ips)\b', # Security systems
    r'(?i)\b(?:access\s+control|authentication|authorization|audit)\b', # Security controls
    # Common network protocols and services
    r'(?i)\b(?:http|https|ftp|ssh|telnet|snmp|dns|dhcp|ntp)\b', # Network protocols
    r'(?i)\b(?:active\s+directory|ad|domain\s+controller|dc)\b', # Directory services
    r'(?i)\b(?:database\s+server|web\s+server|mail\s+server|file\s+server)\b', # Server types
    # Security perimeter terms
    r'(?i)\b(?:perimeter|boundary|edge|border|demarcation)\b', # Network boundaries
    r'(?i)\b(?:inside|outside|external|internal|private|public)\b', # Network sides
]

def blur_faces(image):
    """
    Detects and blurs faces using YuNet, DNN, or Haar Cascade detector
    """
    if use_yunet_detector:
        return blur_faces_yunet(image)
    elif use_dnn_detector:
        return blur_faces_dnn(image)
    else:
        return blur_faces_haar(image)

def blur_faces_yunet(image):
    """
    Most accurate face detection using YuNet model
    """
    h, w = image.shape[:2]
    
    # Set input size for YuNet - try multiple sizes for better small face detection
    input_sizes = [(w, h), (640, 640), (320, 320)]
    all_faces = []
    
    for input_size in input_sizes:
        try:
            face_detector.setInputSize(input_size)
            _, faces = face_detector.detect(image)
            
            if faces is not None:
                for face in faces:
                    x1, y1, face_w, face_h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    confidence = face[14]
                    
                    # Lower confidence threshold for small faces
                    if confidence > 0.3:  # Reduced from 0.6 to catch more faces
                        all_faces.append((x1, y1, face_w, face_h, confidence))
        except:
            continue
    
    # Remove duplicates and sort by confidence
    all_faces.sort(key=lambda x: x[4], reverse=True)
    unique_faces = []
    
    for face in all_faces:
        x, y, w, h, conf = face
        is_duplicate = False
        
        for unique_face in unique_faces:
            ux, uy, uw, uh = unique_face[:4]
            # Check for overlap
            overlap_x = max(0, min(x + w, ux + uw) - max(x, ux))
            overlap_y = max(0, min(y + h, uy + uh) - max(y, uy))
            overlap_area = overlap_x * overlap_y
            face_area = w * h
            
            if overlap_area > face_area * 0.3:
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_faces.append((x, y, w, h))
    
    # Blur detected faces
    for (x, y, w, h) in unique_faces:
        # Add padding
        padding = max(8, min(w, h) // 8)  # Increased padding
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        face_roi = image[y_start:y_end, x_start:x_end]
        
        # Adaptive blur kernel - more aggressive for small faces
        kernel_size = max(26, min(w, h) // 2)  # Increased blur for small faces
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, 41)  # Increased max blur
        
        blurred_face = cv2.GaussianBlur(face_roi, (kernel_size, kernel_size), 0)
        image[y_start:y_end, x_start:x_end] = blurred_face
    
    return image
    
def blur_faces_dnn(image):
    """
    More accurate face detection using DNN model
    """
    h, w = image.shape[:2]
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
    face_net.setInput(blob)
    detections = face_net.forward()
    
    faces_detected = []
    
    # Process detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections (adjust threshold as needed)
        if confidence > 0.3:  # Lower threshold to catch more faces, including small ones
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            
            # Convert to (x, y, width, height) format
            face_w = x2 - x1
            face_h = y2 - y1
            
            # Filter out invalid detections
            if face_w > 0 and face_h > 0:
                faces_detected.append((x1, y1, face_w, face_h, confidence))
    
    # Sort by confidence and remove overlapping detections
    faces_detected.sort(key=lambda x: x[4], reverse=True)  # Sort by confidence
    
    unique_faces = []
    for face in faces_detected:
        x, y, w, h, conf = face
        is_duplicate = False
        
        for unique_face in unique_faces:
            ux, uy, uw, uh = unique_face[:4]
            # Check for overlap
            overlap_x = max(0, min(x + w, ux + uw) - max(x, ux))
            overlap_y = max(0, min(y + h, uy + uh) - max(y, uy))
            overlap_area = overlap_x * overlap_y
            face_area = w * h
            
            if overlap_area > face_area * 0.3:
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_faces.append((x, y, w, h))
    
    # Blur detected faces
    for (x, y, w, h) in unique_faces:
        # Add padding
        padding = max(5, min(w, h) // 10)
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        face_roi = image[y_start:y_end, x_start:x_end]
        
        # Adaptive blur kernel
        kernel_size = max(5, min(w, h) // 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, 33)
        
        blurred_face = cv2.GaussianBlur(face_roi, (kernel_size, kernel_size), 0)
        image[y_start:y_end, x_start:x_end] = blurred_face
    
    return image

def blur_faces_haar(image):
    """
    Fallback face detection using Haar Cascade (original method)
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try multiple detection passes with different parameters for better small face detection
    face_sets = []
    
    # Standard detection
    faces1 = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_sets.append(faces1)
    
    # More sensitive detection for smaller faces (like ID cards)
    faces2 = face_cascade.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=3, minSize=(15, 15), maxSize=(200, 200))
    face_sets.append(faces2)
    
    # Even more aggressive detection
    faces3 = face_cascade.detectMultiScale(gray_image, scaleFactor=1.03, minNeighbors=2, minSize=(10, 10), maxSize=(150, 150))
    face_sets.append(faces3)
    
    # Combine all detections and remove duplicates
    all_faces = []
    for face_set in face_sets:
        for face in face_set:
            all_faces.append(face)

    # Remove overlapping detections (simple approach)
    unique_faces = []
    for face in all_faces:
        x, y, w, h = face
        is_duplicate = False
        for unique_face in unique_faces:
            ux, uy, uw, uh = unique_face
            # Check for significant overlap
            overlap_x = max(0, min(x + w, ux + uw) - max(x, ux))
            overlap_y = max(0, min(y + h, uy + uh) - max(y, uy))
            overlap_area = overlap_x * overlap_y
            face_area = w * h
            if overlap_area > face_area * 0.3:  # 30% overlap threshold
                is_duplicate = True
                break
        if not is_duplicate:
            unique_faces.append(face)
    
    # Blur all detected faces
    for (x, y, w, h) in unique_faces:
        # Add some padding around the detected face for ID cards
        padding = max(5, min(w, h) // 10)
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        face_roi = image[y_start:y_end, x_start:x_end]
        
        # Use smaller blur kernel for small faces
        kernel_size = max(5, min(w, h) // 3)
        if kernel_size % 2 == 0:  # Ensure odd number for Gaussian blur
            kernel_size += 1
        kernel_size = min(kernel_size, 33)  # Cap the blur size
        
        blurred_face = cv2.GaussianBlur(face_roi, (kernel_size, kernel_size), 0)
        image[y_start:y_end, x_start:x_end] = blurred_face
        
    return image

def blur_id_cards(image):
    """
    Detects and blurs ID cards/badges in the image.
    Uses multiple detection approaches for better coverage.
    """
    original_image = image.copy()
    
    # Method 1: Traditional edge-based detection
    image = blur_id_cards_edges(image)
    
    # Method 2: Color-based detection for white cards
    image = blur_white_cards(image)
    
    # Method 3: Template matching approach
    image = blur_rectangular_objects(image)
    
    return image

def blur_white_cards(image):
    """
    Specifically targets white/light colored rectangular objects like ID cards
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create mask for bright areas (white cards)
    _, bright_mask = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)
    
    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in bright areas
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_height, img_width = image.shape[:2]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Look for medium-sized bright areas
        if 1000 < area < (img_width * img_height * 0.2):
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it could be a card
            aspect_ratio = w / h if h > 0 else 0
            if 0.5 < aspect_ratio < 2.5:
                
                # Check if the area is uniformly bright (like a card)
                card_roi = gray_image[y:y+h, x:x+w]
                if card_roi.size > 0:
                    mean_brightness = np.mean(card_roi)
                    brightness_std = np.std(card_roi)
                    
                    # White cards should be bright with some variation (text/photos)
                    if mean_brightness > 160 and 5 < brightness_std < 80:
                        
                        # Apply blur with generous padding
                        padding = max(15, min(w, h) // 8)
                        x_start = max(0, x - padding)
                        y_start = max(0, y - padding)
                        x_end = min(img_width, x + w + padding)
                        y_end = min(img_height, y + h + padding)
                        
                        id_roi = image[y_start:y_end, x_start:x_end]
                        blur_kernel = min(31, max(19, min(w, h) // 5))
                        if blur_kernel % 2 == 0:
                            blur_kernel += 1
                            
                        blurred_id = cv2.GaussianBlur(id_roi, (blur_kernel, blur_kernel), 0)
                        image[y_start:y_end, x_start:x_end] = blurred_id
    
    return image

def blur_rectangular_objects(image):
    """
    Detects rectangular objects that might be cards using shape analysis
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use adaptive threshold to handle varying lighting
    adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_height, img_width = image.shape[:2]

    for contour in contours:
        area = cv2.contourArea(contour)

        if 800 < area < (img_width * img_height * 0.25):

            # Fit a rectangle to the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            # CORRECTED LINE HERE:
            box = np.intp(box)  # Changed from np.int0 to np.intp (or np.int32)

            # Get the bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Check aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if 0.4 < aspect_ratio < 3.0:

                # Check if it's card-sized (not too big, not too small)
                diagonal = np.sqrt(w * w + h * h)
                img_diagonal = np.sqrt(img_width * img_width + img_height * img_height)

                if 0.05 < (diagonal / img_diagonal) < 0.4:  # 5-40% of image diagonal

                    # Apply blur
                    padding = max(12, min(w, h) // 10)
                    x_start = max(0, x - padding)
                    y_start = max(0, y - padding)
                    x_end = min(img_width, x + w + padding)
                    y_end = min(img_height, y + h + padding)

                    id_roi = image[y_start:y_end, x_start:x_end]
                    blur_kernel = min(27, max(15, min(w, h) // 6))
                    if blur_kernel % 2 == 0:
                        blur_kernel += 1

                    blurred_id = cv2.GaussianBlur(id_roi, (blur_kernel, blur_kernel), 0)
                    image[y_start:y_end, x_start:x_end] = blurred_id

    return image

def blur_id_cards_edges(image):
    """
    Original edge-based ID card detection (kept as backup)
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection to find rectangular shapes - try multiple approaches
    edges1 = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    edges2 = cv2.Canny(gray_image, 30, 100, apertureSize=3)  # More sensitive
    
    # Combine edge maps
    edges = cv2.bitwise_or(edges1, edges2)
    
    # Apply morphological operations to connect broken edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get image dimensions for filtering
    img_height, img_width = image.shape[:2]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if 500 < area < (img_width * img_height * 0.4):
            
            epsilon = 0.03 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 3:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                if 0.4 < aspect_ratio < 3.0:
                    bounding_area = w * h
                    fill_ratio = area / bounding_area if bounding_area > 0 else 0
                    
                    if fill_ratio > 0.2:
                        card_roi = gray_image[y:y+h, x:x+w]
                        if card_roi.size > 0:
                            brightness_std = np.std(card_roi)
                            brightness_mean = np.mean(card_roi)
                            
                            if brightness_std > 10 or brightness_mean > 100:
                                padding = max(8, min(w, h) // 15)
                                x_start = max(0, x - padding)
                                y_start = max(0, y - padding)
                                x_end = min(img_width, x + w + padding)
                                y_end = min(img_height, y + h + padding)
                                
                                id_roi = image[y_start:y_end, x_start:x_end]
                                blur_kernel = min(25, max(15, min(w, h) // 6))
                                if blur_kernel % 2 == 0:
                                    blur_kernel += 1
                                    
                                blurred_id = cv2.GaussianBlur(id_roi, (blur_kernel, blur_kernel), 0)
                                image[y_start:y_end, x_start:x_end] = blurred_id
    
    return image

def blur_screens(image):
    """
    Improved screen detection to avoid blurring entire screenshots
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray_image, (7, 7), 0)

    # Use a high threshold to isolate bright areas
    _, thresh = cv2.threshold(blurred_gray, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get image dimensions for size filtering
    img_height, img_width = image.shape[:2]
    img_area = img_height * img_width

    for contour in contours:
        contour_area = cv2.contourArea(contour)

        # Filter contours based on area and shape to find screen-like objects
        # Exclude contours that are too large (likely the entire image/screenshot)
        if 5000 < contour_area < (img_area * 0.8):  # Don't blur if contour is >80% of image
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4: # Check if it's a quadrilateral
                x, y, w, h = cv2.boundingRect(contour)

                # Additional checks to avoid blurring entire image regions
                bbox_area = w * h

                # Skip if bounding box is too large relative to image
                if bbox_area > (img_area * 0.7):
                    continue

                # Skip if bounding box covers most of the image dimensions
                if (w > img_width * 0.8) or (h > img_height * 0.8):
                    continue

                # Add aspect ratio check - screens usually have reasonable aspect ratios
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.3 or aspect_ratio > 5.0:  # Skip very thin or very wide rectangles
                    continue

                screen_roi = image[y:y+h, x:x+w]
                blurred_screen = cv2.GaussianBlur(screen_roi, (23, 23), 10) # Reduced blur for screens
                image[y:y+h, x:x+w] = blurred_screen

    return image

def blur_sensitive_text(image):
    # Get OCR data: text and bounding boxes for each word
    ocr_data = pytesseract.image_to_data(
        image,
        config="--psm 11 --oem 1",  # PSM 11 = sparse text, OEM 1 = LSTM (better handwriting support)
        output_type=pytesseract.Output.DICT
    )
    full_text_str = " ".join(ocr_data['text']).strip()

    # Use spaCy to find sensitive entities
    doc = nlp(full_text_str)

    # Store all sensitive spans (start_char, end_char)
    sensitive_spans = []

    # 1. Add spaCy entities
    for ent in doc.ents:
        if ent.label_ in SENSITIVE_ENTITIES:
            sensitive_spans.append((ent.start_char, ent.end_char))
            # print(f"SpaCy sensitive: '{ent.text}' (Label: {ent.label_})") # Debugging

    # 2. Add Regex pattern matches
    # 2. Add Regex pattern matches
    for pattern in SENSITIVE_REGEX_PATTERNS:
        for match in re.finditer(pattern, full_text_str):
            start, end = match.span()
            sensitive_spans.append((start, end))

    # 2b. Extra fallback regex for handwritten / two-word names
    NAME_FALLBACK = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")
    for match in re.finditer(NAME_FALLBACK, full_text_str):
        sensitive_spans.append(match.span())

    # Convert word-level OCR data to actual character positions in full_text_str
    # This helps map OCR bounding boxes to spaCy/regex sensitive spans
    word_char_spans = []
    current_char_pos = 0
    for word_idx in range(len(ocr_data['text'])):
        word = ocr_data['text'][word_idx]
        if word.strip():
            start_pos = full_text_str.find(word, current_char_pos)
            if start_pos != -1:
                end_pos = start_pos + len(word)
                word_char_spans.append((word_idx, start_pos, end_pos))
                current_char_pos = end_pos # Advance position for next search
            else: # Word not found, likely due to OCR errors or spacing
                word_char_spans.append((word_idx, -1, -1)) # Mark as invalid
        else:
            word_char_spans.append((word_idx, -1, -1)) # Mark as invalid

    # Iterate through OCR detected words and blur if they fall within a sensitive span
    n_boxes = len(ocr_data['level'])
    for i in range(n_boxes):
        if ocr_data['conf'][i] != '-1' and int(ocr_data['conf'][i]) < 10:  # was 30 or 50 earlier
            continue
        # Skip empty OCR results

        # Check if this OCR word's character span overlaps with any sensitive span
        word_idx, word_start_char, word_end_char = word_char_spans[i]

        if word_start_char == -1: # Skip words that couldn't be mapped
            continue

        is_sensitive = False
        for sens_start, sens_end in sensitive_spans:
            # Check for overlap: word starts before sensitive end AND word ends after sensitive start
            if max(word_start_char, sens_start) < min(word_end_char, sens_end):
                is_sensitive = True
                break

        if is_sensitive:
            (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])

            # Ensure coordinates are valid
            if w > 0 and h > 0:
                text_roi = image[y:y+h, x:x+w]
                # Applying a lighter blur for text, as per your request
                blurred_text = cv2.GaussianBlur(text_roi, (45, 45), 30) # Smaller blur for text
                image[y:y+h, x:x+w] = blurred_text

    return image

def detect_flowchart_diagram(image):
    """
    Detects if an image contains flowcharts, network diagrams, or architectural diagrams
    Returns True if it looks like a technical diagram that should be fully blurred
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Look for characteristics common in flowcharts/diagrams:
    
    # 1. Count rectangular shapes (boxes/components)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangular_shapes = 0
    total_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Ignore very small shapes
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Count rectangular-ish shapes (3-6 corners for flexibility)
            if 3 <= len(approx) <= 6:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Typical diagram components are roughly rectangular
                if 0.3 < aspect_ratio < 4.0 and area > 1000:
                    rectangular_shapes += 1
                    total_area += area
    
    # 2. Look for connecting lines (edges that are long and straight)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    line_count = len(lines) if lines is not None else 0
    
    # 3. Check color distribution - diagrams often have distinct colored regions
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Count distinct color regions
    color_regions = 0
    colors_to_check = [
        ([0, 50, 50], [10, 255, 255]),    # Red range
        ([25, 50, 50], [35, 255, 255]),   # Yellow range  
        ([45, 50, 50], [75, 255, 255]),   # Green range
        ([100, 50, 50], [130, 255, 255]), # Blue range
        ([0, 0, 200], [180, 30, 255])     # Light colors (near white)
    ]
    
    for (lower, upper) in colors_to_check:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) > (w * h * 0.05):  # If color covers >5% of image
            color_regions += 1
    
    # 4. Text density check - diagrams have labels but not dense text
    try:
        # Quick OCR check for text characteristics
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        text_boxes = [text for text in ocr_data['text'] if text.strip()]
        text_density = len(text_boxes) / (w * h / 10000)  # Text boxes per 10k pixels
    except:
        text_density = 0
    
    # Decision criteria for flowchart/diagram detection:
    diagram_score = 0
    
    # Multiple rectangular shapes suggest diagram components
    if rectangular_shapes >= 4:
        diagram_score += 2
    elif rectangular_shapes >= 2:
        diagram_score += 1
        
    # Many lines suggest connections between components
    if line_count >= 10:
        diagram_score += 2
    elif line_count >= 5:
        diagram_score += 1
        
    # Multiple distinct colored regions (like your network diagram)
    if color_regions >= 3:
        diagram_score += 2
    elif color_regions >= 2:
        diagram_score += 1
        
    # Moderate text density (labels but not document-heavy)
    if 0.1 < text_density < 2.0:
        diagram_score += 1
        
    # High coverage of geometric shapes relative to image size
    shape_coverage = total_area / (w * h) if (w * h) > 0 else 0
    if shape_coverage > 0.3:
        diagram_score += 1
        
    # Debug info (uncomment to see detection details)
    # print(f"Flowchart detection - Rectangles: {rectangular_shapes}, Lines: {line_count}, "
    #       f"Colors: {color_regions}, Text density: {text_density:.2f}, Score: {diagram_score}")
    
    # If score is high enough, treat as flowchart/diagram
    return diagram_score >= 4

def blur_entire_image(image, blur_intensity="medium"):
    """
    Blurs the entire image for flowcharts/diagrams
    """
    if blur_intensity == "light":
        kernel_size = 15
    elif blur_intensity == "medium":
        kernel_size = 25
    else:  # heavy
        kernel_size = 35
        
    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred
    # Get OCR data: text and bounding boxes for each word
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    full_text_str = " ".join(ocr_data['text']).strip()
    
    # Use spaCy to find sensitive entities
    doc = nlp(full_text_str)
    
    # Store all sensitive spans (start_char, end_char)
    sensitive_spans = []

    # 1. Add spaCy entities
    for ent in doc.ents:
        if ent.label_ in SENSITIVE_ENTITIES:
            sensitive_spans.append((ent.start_char, ent.end_char))
            # print(f"SpaCy sensitive: '{ent.text}' (Label: {ent.label_})") # Debugging

    # 2. Add Regex pattern matches
    for pattern in SENSITIVE_REGEX_PATTERNS:
        for match in re.finditer(pattern, full_text_str):
            start, end = match.span()
            # print(f"Regex sensitive: '{match.group(0)}'") # Debugging
            sensitive_spans.append((start, end))

    # Convert word-level OCR data to actual character positions in full_text_str
    # This helps map OCR bounding boxes to spaCy/regex sensitive spans
    word_char_spans = []
    current_char_pos = 0
    for word_idx in range(len(ocr_data['text'])):
        word = ocr_data['text'][word_idx]
        if word.strip():
            start_pos = full_text_str.find(word, current_char_pos)
            if start_pos != -1:
                end_pos = start_pos + len(word)
                word_char_spans.append((word_idx, start_pos, end_pos))
                current_char_pos = end_pos # Advance position for next search
            else: # Word not found, likely due to OCR errors or spacing
                word_char_spans.append((word_idx, -1, -1)) # Mark as invalid
        else:
            word_char_spans.append((word_idx, -1, -1)) # Mark as invalid

    # Iterate through OCR detected words and blur if they fall within a sensitive span
    n_boxes = len(ocr_data['level'])
    for i in range(n_boxes):
        if ocr_data['text'][i].strip() == "":
            continue # Skip empty OCR results

        # Check if this OCR word's character span overlaps with any sensitive span
        word_idx, word_start_char, word_end_char = word_char_spans[i]

        if word_start_char == -1: # Skip words that couldn't be mapped
            continue

        is_sensitive = False
        for sens_start, sens_end in sensitive_spans:
            # Check for overlap: word starts before sensitive end AND word ends after sensitive start
            if max(word_start_char, sens_start) < min(word_end_char, sens_end):
                is_sensitive = True
                break
        
        if is_sensitive:
            (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
            
            # Ensure coordinates are valid
            if w > 0 and h > 0:
                text_roi = image[y:y+h, x:x+w]
                # Applying a lighter blur for text, as per your request
                blurred_text = cv2.GaussianBlur(text_roi, (15, 15), 5) # Smaller blur for text
                image[y:y+h, x:x+w] = blurred_text
                    
    return image

def is_likely_screenshot(image_path):
    """
    Simple heuristic to detect if an image is likely a screenshot.
    You can expand this logic based on your needs.
    """
    # Check file name for common screenshot indicators
    filename = os.path.basename(image_path).lower()
    screenshot_keywords = ['screenshot', 'screen', 'capture', 'shot']
    
    for keyword in screenshot_keywords:
        if keyword in filename:
            return True
    
    return False

def process_image(image_path, output_dir):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read image {image_path}")
        return

    blurred_image = original_image.copy()

    print(f"Processing {image_path}...")

    # Apply blurring functions
    print("1. Blurring faces...")
    blurred_image = blur_faces(blurred_image)

    print("2. Blurring screens...")
    blurred_image = blur_screens(blurred_image)

    print("3. Blurring sensitive text...")
    blurred_image = blur_sensitive_text(blurred_image)

    # Save and display the result
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_blurred{ext}")
    cv2.imwrite(output_path, blurred_image)
    print(f"-----> Successfully saved processed image to {output_path} <-----")

    # Display images for review (optional)
    # cv2.imshow("Original Image", original_image)
    # cv2.imshow("Blurred Image", blurred_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) != 3:
        print("Usage: python blur_tool.py <input_image_path> <output_directory>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_folder = sys.argv[2]
    
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' not found.")
    else:
        print(f"Running image processing for: {os.path.basename(input_path)}")
        process_image(input_path, output_folder)
