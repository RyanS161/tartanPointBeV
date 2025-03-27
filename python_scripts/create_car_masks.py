import os
from PIL import Image

# Define the input directory and output directories
input_dir = "/Users/ryanslocum/Documents/current_courses/PLR/repos/tartanPointBeV/python_scripts/output/sem/max_ground"
color_mask_dir = "/Users/ryanslocum/Documents/current_courses/PLR/repos/tartanPointBeV/python_scripts/output/sem/max_ground/car_masks"
transparent_mask_dir = "/Users/ryanslocum/Documents/current_courses/PLR/repos/tartanPointBeV/python_scripts/output/sem/max_ground/valid_masks"

# Ensure output directories exist
os.makedirs(color_mask_dir, exist_ok=True)
os.makedirs(transparent_mask_dir, exist_ok=True)

# Define the target color for the binary mask (R, G, B, A)
target_color = (85, 230, 119) 

def create_binary_masks(image_path):
    # Open the image
    image = Image.open(image_path).convert("RGBA")
    pixels = image.load()

    # Create blank images for the masks
    color_mask = Image.new("1", image.size)  # Binary mask for the target color
    transparent_mask = Image.new("1", image.size)  # Binary mask for transparency

    color_mask_pixels = color_mask.load()
    transparent_mask_pixels = transparent_mask.load()
    # Get all unique colors in the image
    # unique_colors = set()
    # for y in range(image.size[1]):
    #     for x in range(image.size[0]):
    #         unique_colors.add(pixels[x, y])
    # unique_colors = unique_colors
    # Iterate through each pixel
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            r, g, b, a = pixels[x, y]

            # Check for the target color
            color_mask_pixels[x, y] = 1 if (r, g, b) == target_color else 0

            # Check for transparency
            transparent_mask_pixels[x, y] = 0 if a == 0 else 1

    return color_mask, transparent_mask

# Process all images in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png')):
        image_path = os.path.join(input_dir, filename)

        # Create the binary masks
        color_mask, transparent_mask = create_binary_masks(image_path)

        # Save the masks
        color_mask.save(os.path.join(color_mask_dir, f"{os.path.splitext(filename)[0]}.png"))
        transparent_mask.save(os.path.join(transparent_mask_dir, f"{os.path.splitext(filename)[0]}.png"))

print("Masks created and saved successfully.")