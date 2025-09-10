"""
Basic Image Manipulation: Loading, Displaying, Resizing, Cropping, Saving

Theory:
- Loading: Read image from file into a NumPy array.
- Displaying: Use libraries like matplotlib or OpenCV's imshow.
- Resizing: Change image dimensions, often with interpolation (e.g., bilinear).
- Cropping: Extract a region of interest (ROI).
- Saving: Write the array back to a file.

Math:
- Resizing: Interpolation formulas, e.g., bilinear: weighted average of neighboring pixels.
- Coordinates: Image coordinates start from top-left (0,0).

Implementation: Using OpenCV for most operations, matplotlib for display.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """
    Load an image from file.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return None
    return image

def display_image(image, title="Image"):
    """
    Display an image using matplotlib.
    """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def resize_image(image, width, height):
    """
    Resize image to specified dimensions.
    """
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized

def crop_image(image, x, y, w, h):
    """
    Crop a region from the image.
    x, y: top-left corner
    w, h: width and height
    """
    cropped = image[y:y+h, x:x+w]
    return cropped

def save_image(image, output_path):
    """
    Save image to file.
    """
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"  # Update this
    image = load_image(image_path)
    if image is not None:
        display_image(image, "Original")

        # Resize
        resized = resize_image(image, 300, 300)
        display_image(resized, "Resized")

        # Crop
        cropped = crop_image(image, 50, 50, 200, 200)
        display_image(cropped, "Cropped")

        # Save
        save_image(resized, "resized_image.jpg")
