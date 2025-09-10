"""
2D Convolution from Scratch

Theory:
- Convolution is a fundamental operation in image processing that applies a kernel (filter) to an image.
- It slides the kernel over the image, computing a weighted sum at each position.
- Used for blurring, sharpening, edge detection, etc.

Math:
- For a kernel K of size (m,n), the convolution at position (i,j) is:
  output[i,j] = sum_{x=0}^{m-1} sum_{y=0}^{n-1} input[i+x, j+y] * K[x,y]
- Padding and stride can be applied to control output size.

Implementation: NumPy-based convolution without OpenCV.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def convolve2d(image, kernel, padding=0, stride=1):
    """
    Perform 2D convolution on a grayscale image.
    """
    # Get dimensions
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    # Calculate output dimensions
    output_h = (image_h + 2 * padding - kernel_h) // stride + 1
    output_w = (image_w + 2 * padding - kernel_w) // stride + 1

    # Pad the image
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)

    # Initialize output
    output = np.zeros((output_h, output_w))

    # Perform convolution
    for i in range(output_h):
        for j in range(output_w):
            region = padded_image[i*stride:i*stride+kernel_h, j*stride:j*stride+kernel_w]
            output[i, j] = np.sum(region * kernel)

    return output

def apply_convolution(image_path, kernel, title="Convolved Image"):
    """
    Load image, apply convolution, display and save.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return

    # Apply convolution
    convolved = convolve2d(image, kernel)

    # Normalize for display
    convolved_norm = cv2.normalize(convolved, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Display
    plt.imshow(convolved_norm, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

    return convolved

if __name__ == "__main__":
    # Example: Sharpening kernel
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])

    image_path = "../images/ganpati_bappa.jpg"  # Update path
    apply_convolution(image_path, sharpening_kernel, "Sharpened Image")
