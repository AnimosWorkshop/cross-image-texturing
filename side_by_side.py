from PIL import Image

def combine_images_vertically(image_path1, image_path2):
    """
    Combine two images vertically using PIL.
    
    Parameters:
    image_path1 (str): Path to the first image
    image_path2 (str): Path to the second image
    
    Returns:
    PIL.Image: Combined image
    """
    # Open both images
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)
    
    # Get dimensions
    width1, height1 = img1.size
    width2, height2 = img2.size
    
    # Set the width of the final image (use the max width)
    final_width = max(width1, width2)
    
    # If images have different widths, resize the smaller one
    if width1 != width2:
        if width1 < width2:
            new_height1 = int(height1 * (final_width / width1))
            img1 = img1.resize((final_width, new_height1), Image.Resampling.LANCZOS)
        else:
            new_height2 = int(height2 * (final_width / width2))
            img2 = img2.resize((final_width, new_height2), Image.Resampling.LANCZOS)
    
    # Update heights after potential resizing
    width1, height1 = img1.size
    width2, height2 = img2.size
    
    # Create a new image with the combined height
    combined_height = height1 + height2
    combined_image = Image.new('RGB', (final_width, combined_height))
    
    # Paste the images
    combined_image.paste(img1, (0, 0))
    combined_image.paste(img2, (0, height1))
    
    return combined_image

# Your image paths
image1_path = "/home/ML_courses/03683533_2024/lidor_yael_snir/lidor_only/cross-image-texturing/data/MVD_30Jan2025-091206/results/textured_views_rgb.jpg"
image2_path = "/home/ML_courses/03683533_2024/lidor_yael_snir/syncmvd/SyncMVD/data/face/MVD_25Dec2024-113913/results/textured_views_rgb.jpg"

# Combine the images
combined_img = combine_images_vertically(image1_path, image2_path)

# Display the combined image
combined_img.show()

# Optionally, save the combined image
# combined_img.save('combined_image.jpg')