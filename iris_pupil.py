import cv2
import os
import numpy as np

# Directory to store cropped eye images
input_dir = "D:\\irispupilfinal\\eyesfinal"
output_dir = "enhanced_eyes"
os.makedirs(output_dir, exist_ok=True)

def remove_specular_reflection(image):
    """
    Removes specular reflections from the eye image using inpainting.
    """
    # Threshold to create a mask for bright spots (specular reflections)
    _, mask = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)  # Adjust threshold as needed
    
    # Dilate the mask to cover surrounding regions of reflections
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Inpaint the regions covered by the mask
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return inpainted_image

def enhance_image(image_path, output_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 1: Remove Specular Reflections
    no_reflection_image = remove_specular_reflection(image)

    # Step 2: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(no_reflection_image)

    # Step 3: Apply Bilateral Filtering for noise reduction while preserving edges
    bilateral_filtered_image = cv2.bilateralFilter(clahe_image, d=15, sigmaColor=100, sigmaSpace=100)

    # Step 4: Apply Sharpening
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5.5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(bilateral_filtered_image, -1, sharpening_kernel)

    # Save the enhanced image
    cv2.imwrite(output_path, sharpened_image)

# Process each image in the detected_eyes directory
for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)

    # Skip if it's not an image file
    if not (filename.lower().endswith('.jpeg') or filename.lower().endswith('.png')):
        continue

    # Generate output file path
    output_path = os.path.join(output_dir, filename)

    # Enhance the image
    enhance_image(input_path, output_path)
    print(f"Enhanced image saved to {output_path}")

print("All images enhanced and saved in the 'enhanced_eyes' directory.")