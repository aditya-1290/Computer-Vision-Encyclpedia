"""
OCR and Document Analysis: Extract text from images using Tesseract.

Implementation uses pytesseract for OCR and OpenCV for preprocessing.

Theory:
- OCR: Optical Character Recognition.
- Preprocessing: Enhance image for better text extraction.
- Document analysis: Layout analysis, text extraction.

Math: Binarization: threshold = (min + max) / 2

Reference:
- Tesseract OCR documentation
"""

import cv2
import pytesseract
import numpy as np
import re
from collections import defaultdict

def enhanced_preprocess_image(image, preprocess_steps=['grayscale', 'denoise', 'threshold']):
    """
    Enhanced preprocessing with multiple steps
    """
    processed = image.copy()
    
    for step in preprocess_steps:
        if step == 'grayscale':
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        elif step == 'denoise':
            processed = cv2.fastNlMeansDenoising(processed)
        elif step == 'threshold':
            processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
        elif step == 'deskew':
            processed = deskew_image(processed)
        elif step == 'remove_noise':
            processed = remove_noise(processed)
        elif step == 'dilate':
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.dilate(processed, kernel, iterations=1)
        elif step == 'erode':
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.erode(processed, kernel, iterations=1)
        elif step == 'opening':
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        elif step == 'closing':
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    
    return processed

def deskew_image(image):
    """
    Deskew image based on text orientation
    """
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), 
                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def remove_noise(image):
    """
    Remove noise from image using morphological operations
    """
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image

def extract_text_with_ocr(image, lang='eng', config='--psm 6', preprocess_steps=None):
    """
    Enhanced text extraction with language support and preprocessing options
    """
    if preprocess_steps:
        image = enhanced_preprocess_image(image, preprocess_steps)
    
    # Custom OCR configuration based on content type
    text = pytesseract.image_to_string(image, lang=lang, config=config)
    return text

def detect_document_structure(image, lang='eng'):
    """
    Detect document structure (paragraphs, headings, etc.)
    """
    # Get OCR data with detailed information
    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    
    # Group text by blocks
    blocks = defaultdict(list)
    n_boxes = len(data['text'])
    
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:
            block_num = data['block_num'][i]
            blocks[block_num].append({
                'text': data['text'][i],
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i],
                'conf': data['conf'][i]
            })
    
    # Analyze block structure
    document_structure = []
    for block_id, elements in blocks.items():
        # Calculate block boundaries
        left = min(e['left'] for e in elements)
        top = min(e['top'] for e in elements)
        right = max(e['left'] + e['width'] for e in elements)
        bottom = max(e['top'] + e['height'] for e in elements)
        width = right - left
        height = bottom - top
        
        # Combine text in block
        text = ' '.join(e['text'] for e in elements if e['text'].strip())
        
        # Estimate font size (average height)
        avg_height = sum(e['height'] for e in elements) / len(elements)
        
        # Classify block type based on heuristics
        if avg_height > 30 and text.isupper():
            block_type = 'heading'
        elif avg_height > 25:
            block_type = 'subheading'
        elif len(text) > 200:
            block_type = 'paragraph'
        else:
            block_type = 'text'
        
        document_structure.append({
            'type': block_type,
            'text': text,
            'position': (left, top, width, height),
            'font_size': avg_height
        })
    
    return document_structure

def extract_tables(image, lang='eng'):
    """
    Extract tables from image
    """
    # Use special OCR config for tables
    table_config = '--psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-()%$ '
    data = pytesseract.image_to_data(image, lang=lang, config=table_config, 
                                   output_type=pytesseract.Output.DICT)
    
    # Group text by rows based on y-position
    rows = defaultdict(list)
    n_boxes = len(data['text'])
    
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60 and data['text'][i].strip():
            # Group by similar y positions (within threshold)
            y = data['top'][i]
            row_key = round(y / 10) * 10  # Group by 10px increments
            rows[row_key].append({
                'text': data['text'][i],
                'left': data['left'][i],
                'width': data['width'][i]
            })
    
    # Sort rows by y-position and cells by x-position
    sorted_rows = sorted(rows.items(), key=lambda x: x[0])
    table = []
    
    for y, cells in sorted_rows:
        sorted_cells = sorted(cells, key=lambda x: x['left'])
        table.append([cell['text'] for cell in sorted_cells])
    
    return table

def analyze_document_advanced(image, lang='eng', detect_tables=True):
    """
    Advanced document analysis with structure detection
    """
    # Preprocess image
    processed = enhanced_preprocess_image(image, ['grayscale', 'denoise', 'threshold'])
    
    # Extract basic text
    text = extract_text_with_ocr(processed, lang)
    
    # Detect document structure
    structure = detect_document_structure(processed, lang)
    
    # Extract tables if requested
    tables = []
    if detect_tables:
        tables = extract_tables(processed, lang)
    
    # Draw analysis on image
    output_image = image.copy()
    for block in structure:
        left, top, width, height = block['position']
        
        # Color code by block type
        if block['type'] == 'heading':
            color = (0, 0, 255)  # Red
        elif block['type'] == 'subheading':
            color = (0, 165, 255)  # Orange
        elif block['type'] == 'paragraph':
            color = (0, 255, 0)  # Green
        else:
            color = (255, 0, 0)  # Blue
        
        cv2.rectangle(output_image, (left, top), (left + width, top + height), color, 2)
        cv2.putText(output_image, block['type'], (left, top - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return {
        'full_text': text,
        'structure': structure,
        'tables': tables,
        'annotated_image': output_image
    }
