from PIL import Image
import numpy as np

def calculate_iou(image1_path, image2_path):
    """
    Calculates the Intersection over Union (IoU) for black and white pixels between two binary images.

    Parameters:
    - image1_path: Path to the first image
    - image2_path: Path to the second image

    Returns:
    - A dictionary with IoU values for black and white pixels
    """
    # Load images and convert to binary (grayscale mode)
    image1 = Image.open(image1_path).convert("L")
    image2 = Image.open(image2_path).convert("L")

    # Convert images to numpy arrays with binary values (0 for black, 255 for white)
    array1 = np.array(image1)
    array2 = np.array(image2)

    # Create binary masks for black (0) and white (255) pixels
    black_mask1 = array1 == 0
    black_mask2 = array2 == 0

    white_mask1 = array1 == 255
    white_mask2 = array2 == 255

    # Calculate IoU for black pixels
    black_intersection = np.logical_and(black_mask1, black_mask2).sum()
    black_union = np.logical_or(black_mask1, black_mask2).sum()
    black_iou = black_intersection / black_union if black_union > 0 else 0.0

    # Calculate IoU for white pixels
    white_intersection = np.logical_and(white_mask1, white_mask2).sum()
    white_union = np.logical_or(white_mask1, white_mask2).sum()
    white_iou = white_intersection / white_union if white_union > 0 else 0.0
    
    
    total_intersection = black_intersection + white_intersection
    total_union = black_union + white_union
    total_iou = total_intersection / total_union if total_union > 0 else 0.0

    return {"black_iou": black_iou, "white_iou": white_iou, "total_iou": total_iou}

# Example usage
result = calculate_iou("C:/Users/Karnav/Pictures/Screenshots/truth.png", "C:/Users/Karnav/Pictures/Screenshots/edited2.png")
print(f"IoU for Black Pixels: {result['black_iou']:.4f}")
print(f"IoU for White Pixels: {result['white_iou']:.4f}")
print(f"Total IoU: {result['total_iou']:.4f}")
