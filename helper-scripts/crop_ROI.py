import cv2
import os
import numpy as np

# Create output directory
output_dir = "../cropped_frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get input directory path
input_dir = input("Enter the directory path containing images: ")

# Get list of image files
image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
image_files = [f for f in os.listdir(input_dir)
               if any(f.lower().endswith(ext) for ext in image_extensions)]

if not image_files:
    print("No image files found in the directory")
    exit()

# Load first image to select ROI
first_image_path = os.path.join(input_dir, image_files[0])
first_img = cv2.imread(first_image_path)
original_height, original_width = first_img.shape[:2]

# Resize image for display (max 900p height to fit 1080p screen with UI elements)
max_display_height = 900
scale_factor = 1.0

if original_height > max_display_height:
    scale_factor = max_display_height / original_height
    display_width = int(original_width * scale_factor)
    display_height = int(original_height * scale_factor)
    display_img = cv2.resize(first_img, (display_width, display_height))
else:
    display_img = first_img.copy()

# Let user select ROI on resized image
print(f"Select ROI on image: {image_files[0]}")
print(f"Original size: {original_width}x{original_height}")
print(f"Display size: {display_img.shape[1]}x{display_img.shape[0]}")

roi = cv2.selectROI("Select ROI (Press Enter when done, ESC to cancel)", display_img, False)
cv2.destroyAllWindows()

# Check if ROI was selected
if roi[2] == 0 or roi[3] == 0:
    print("No ROI selected. Exiting.")
    exit()

# Scale ROI coordinates back to original image size
x = int(roi[0] / scale_factor)
y = int(roi[1] / scale_factor)
w = int(roi[2] / scale_factor)
h = int(roi[3] / scale_factor)

print(f"ROI in original image: x={x}, y={y}, w={w}, h={h}")

# Process all images with the same ROI
for image_file in image_files:
    # Load image
    img_path = os.path.join(input_dir, image_file)
    img = cv2.imread(img_path)

    # Check if ROI is within image bounds
    if y + h > img.shape[0] or x + w > img.shape[1]:
        print(f"Skipping {image_file}: ROI exceeds image boundaries")
        continue

    # Crop using the same ROI
    cropped_img = img[y:y + h, x:x + w]

    # Save cropped image
    output_path = os.path.join(output_dir, f"cropped_{image_file}")
    cv2.imwrite(output_path, cropped_img)
    print(f"Cropped: {image_file} - Output size: {cropped_img.shape[1]}x{cropped_img.shape[0]}")

print(f"\nAll images cropped and saved to '{output_dir}' folder")