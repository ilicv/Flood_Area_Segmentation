import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import cv2

# --- Utilities ---
def load_image(path, target_size):
    """
    Load an image from disk and resize to target_size, returning a normalized float32 array.
    """
    img = tf.keras.preprocessing.image.load_img(path, target_size=(target_size, target_size))
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return arr  # shape: (H, W, 3)


def overlay_mask(image, mask, alpha=0.4):
    """
    Overlay a binary mask on an RGB image (assumes image in range [0,1]).
    """
    # Convert image to uint8
    img_uint8 = (image * 255).astype(np.uint8)
    mask_uint8 = (mask * 255).astype(np.uint8)
    colored_mask = np.zeros_like(img_uint8)
    # Red channel
    colored_mask[:, :, 0] = mask_uint8
    # Blend mask onto image
    overlay = cv2.addWeighted(colored_mask, alpha, img_uint8, 1 - alpha, 0)
    result = img_uint8.copy()
    # Where mask present, use overlay
    result[mask.astype(bool)] = overlay[mask.astype(bool)]
    return result

# --- Main inference script ---
def main():
    parser = argparse.ArgumentParser(description='Predict flood areas with a trained UNet model')
    parser.add_argument('--model', type=str, default='best_flood_unet_tf.h5',
                        help='Path to trained model (.h5)')
    parser.add_argument('--input-dir', type=str, default='Image',
                        help='Directory with input images')
    parser.add_argument('--output-dir', type=str, default='predicted_results',
                        help='Directory to save overlayed predictions')
    parser.add_argument('--ext', type=str, default='.jpg',
                        help='Image file extension (e.g. .jpg, .png)')
    parser.add_argument('--size', type=int, default=256,
                        help='Input size for model')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for mask')
    args = parser.parse_args()

    # Load the model
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = tf.keras.models.load_model(str(model_path), compile=False)

    # Prepare directories
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    # Iterate over input images
    for img_file in sorted(input_path.glob(f'*{args.ext}')):
        # Load and preprocess
        img = load_image(str(img_file), args.size)
        # Predict mask
        pred = model.predict(np.expand_dims(img, 0))[0, :, :, 0]
        mask = (pred > args.threshold).astype(np.uint8)

        # Overlay and save
        result = overlay_mask(img, mask, alpha=0.4)
        save_path = output_path / img_file.name
        # Convert RGB to BGR for cv2
        cv2.imwrite(str(save_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"Saved: {save_path}")

if __name__ == '__main__':
    main()
