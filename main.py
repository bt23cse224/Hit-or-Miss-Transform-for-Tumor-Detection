import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_hit_or_miss(image_path, output_dir):
    """
    Applies the Hit-or-Miss transform to an image to detect tumor-like shapes.
    """
    # --- Step 1: Load and Convert Image to Binary ---
    print(f"Processing image: {image_path}")
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image_gray is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Convert to a binary image. We assume the tumor is a bright object.
    # The threshold value (e.g., 150) is crucial and may need tuning for different images.
    # Foreground (tumor) will be 255 (white), background will be 0 (black).
    _, binary_image = cv2.threshold(image_gray, 150, 255, cv2.THRESH_BINARY)

    # --- Step 2: Define Structuring Elements (Kernels) ---
    # B1 (Hit Kernel): Should look like the shape of the tumor. We'll use an ellipse.
    # This kernel will detect shapes that are at least 20x20 pixels.
    hit_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

    # B2 (Miss Kernel): A ring around the hit kernel. This ensures there is a clear
    # background surrounding the tumor shape.
    # We create a larger ellipse and "carve out" the center.
    miss_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    miss_kernel[10:30, 10:30] = 0 # Carve out the central 20x20 area

    # --- Step 3: Apply Erosion on the Image (The "Hit" Pass) ---
    # This finds all the spots where the hit_kernel can fit inside the foreground.
    erosion_hit = cv2.erode(binary_image, hit_kernel, iterations=1)

    # --- Step 4: Apply Erosion on the Complement Image (The "Miss" Pass) ---
    # First, get the complement of the binary image.
    complement_image = cv2.bitwise_not(binary_image)
    # This finds all the spots where the miss_kernel can fit inside the background.
    erosion_miss = cv2.erode(complement_image, miss_kernel, iterations=1)

    # --- Step 5: Intersect the Images ---
    # The result is the intersection (logical AND) of the two erosions.
    # This gives us the exact locations where the foreground and background patterns were met.
    hit_or_miss_result = cv2.bitwise_and(erosion_hit, erosion_miss)

    # --- Visualization and Saving ---
    base_filename = os.path.basename(image_path)
    titles = ['Original', 'Structured element','Image after Erosion']
    images = [image_gray, binary_image  , erosion_miss]

    plt.figure(figsize=(20, 5))
    for i in range(len(images)):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    
    plt.suptitle(f'Hit-or-Miss Transform for: {base_filename}', fontsize=16)
    plt.show()

    # Save the final result
    output_path = os.path.join(output_dir, f"result_{base_filename}")
    cv2.imwrite(output_path, hit_or_miss_result)
    print(f"Result saved to: {output_path}\n")



# --- Main execution block ---
if __name__ == "__main__":
    # Define paths
    imagepath = r"Tr-me_0019.jpg"
    result_dir = "results"
    apply_hit_or_miss(imagepath,result_dir)