import os
import numpy as np
import torch
from pathlib import Path
import cv2
from PIL import Image

# Import your PyTorch U-Net definition
from train_flood_area_segmentation_unet_pytorch import UNet

# --- Utilities ---
def load_image(path, target_size):
    """
    Load an image from disk, resize to (target_size, target_size),
    and return a normalized float32 NumPy array in [0,1].
    """
    img = Image.open(path).convert('RGB')
    img = img.resize((target_size, target_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr  # shape: (H, W, 3)


def overlay_mask(image, mask, alpha=0.4):
    """
    Overlay a binary mask on an RGB image.
    image: HxWx3 float array in [0,1]
    mask:  HxW uint8 array {0,1}
    """
    img_uint8 = (image * 255).astype(np.uint8)
    mask_uint8 = (mask * 255).astype(np.uint8)
    colored_mask = np.zeros_like(img_uint8)
    # Red channel overlay
    colored_mask[..., 0] = mask_uint8
    # Blend colored mask
    overlay = cv2.addWeighted(colored_mask, alpha, img_uint8, 1 - alpha, 0)
    # Wherever mask, use overlay
    combined = img_uint8.copy()
    combined[mask.astype(bool)] = overlay[mask.astype(bool)]
    return combined

# --- Main Inference Script ---
def main(
    model_path: str = 'best_flood_unet.pth',
    input_dir: str = 'Image',
    output_dir: str = 'predicted_results',
    img_ext: str = '.jpg',
    target_size: int = 256,
    threshold: float = 0.5
):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize and load model
    model = UNet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Prepare paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each image
    for img_file in input_path.glob(f'*{img_ext}'):
        # Load and preprocess
        img_arr = load_image(str(img_file), target_size)
        # Convert to tensor: BxCxHxW
        tensor = torch.from_numpy(img_arr.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            pred = model(tensor)
        pred = pred.squeeze().cpu().numpy()  # HxW

        # Binarize mask
        mask = (pred > threshold).astype(np.uint8)

        # Overlay and save
        result = overlay_mask(img_arr, mask)
        # Save with _pt suffix before extension
        suffix_name = f"{img_file.stem}_pt{img_file.suffix}"
        save_path = output_path / suffix_name
        # Save as JPG (OpenCV expects BGR)
        cv2.imwrite(str(save_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"Saved: {save_path}")

if __name__ == '__main__':
    main()
