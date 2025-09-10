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

def preprocess_image(image):
    """
    Preprocess image for OCR.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Thresholding
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text(image):
    """
    Extract text from image using Tesseract.
    """
    preprocessed = preprocess_image(image)
    text = pytesseract.image_to_string(preprocessed)
    return text

def analyze_document(image):
    """
    Analyze document layout and extract text blocks.
    """
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:  # Confidence threshold
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = data['text'][i]
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image, data

if __name__ == "__main__":
    # Assume image loaded
    # text = extract_text(image)
    # analyzed_image, data = analyze_document(image)
    print("OCR and document analysis functions defined.")
