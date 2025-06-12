import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Suppress TensorFlow INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Enable dynamic GPU memory growth to avoid OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# -------- Custom Losses & Metrics --------
def dice_loss(y_true, y_pred, eps=1e-6):
    axes = (1,2,3)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    sums = tf.reduce_sum(y_true + y_pred, axis=axes)
    dice = (2 * intersection + eps) / (sums + eps)
    return 1 - tf.reduce_mean(dice)

bce_loss = tf.keras.losses.BinaryCrossentropy()
def composite_loss(y_true, y_pred):
    return bce_loss(y_true, y_pred) + dice_loss(y_true, y_pred)


def iou_metric(y_true, y_pred, threshold=0.5, eps=1e-6):
    pred_bin = tf.cast(y_pred > threshold, tf.float32)
    axes = (1,2,3)
    inter = tf.reduce_sum(y_true * pred_bin, axis=axes)
    union = tf.reduce_sum(y_true + pred_bin, axis=axes) - inter
    iou = (inter + eps) / (union + eps)
    return tf.reduce_mean(iou)

# -------- U-Net Model --------
def build_unet(input_shape=(256,256,3)):
    inputs = Input(input_shape)
    def conv_block(x, filters):
        x = Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = Conv2D(filters, 3, activation='relu', padding='same')(x)
        return x

    c1 = conv_block(inputs, 64)
    p1 = MaxPooling2D()(c1)
    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D()(c2)
    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D()(c3)
    c4 = conv_block(p3, 512)
    p4 = MaxPooling2D()(c4)
    b  = conv_block(p4, 1024)

    def up_block(x, skip, filters):
        x = Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
        x = Concatenate()([x, skip])
        return conv_block(x, filters)

    u4 = up_block(b, c4, 512)
    u3 = up_block(u4, c3, 256)
    u2 = up_block(u3, c2, 128)
    u1 = up_block(u2, c1, 64)

    outputs = Conv2D(1, 1, activation='sigmoid')(u1)
    return Model(inputs, outputs)

# -------- Data Loading --------
def load_data(base: Path, size=256):
    img_dir = base / 'Image'
    mask_dir = base / 'Mask'
    if not img_dir.exists() or not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not mask_dir.exists() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    exts = {'.png', '.jpg', '.jpeg'}
    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    if not img_paths:
        raise FileNotFoundError(f"No image files found in {img_dir}")

    X, Y = [], []
    for img_path in img_paths:
        stem = img_path.stem
        mask_candidates = [p for p in mask_dir.iterdir() if p.stem == stem and p.suffix.lower() in exts]
        if not mask_candidates:
            raise FileNotFoundError(f"No mask file found for image {stem}")
        mask_path = mask_candidates[0]

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        img = cv2.resize(img, (size, size)) / 255.0
        mask = cv2.resize(mask, (size, size))
        mask = (mask > 127).astype(np.float32)[..., None]
        X.append(img)
        Y.append(mask)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# -------- Main Training Workflow --------
if __name__ == '__main__':
    base = Path(__file__).parent
    X, Y = load_data(base)
    N = len(X)
    print(f"Dataset loaded. Total samples: {N}")
    print(f"Train/Val split: {int(0.8*N)}/{N - int(0.8*N)}")

    idx = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(idx)
    split = int(0.8 * N)
    train_idx, val_idx = idx[:split], idx[split:]
    x_train, x_val = X[train_idx], X[val_idx]
    y_train, y_val = Y[train_idx], Y[val_idx]

    model = build_unet((256,256,3))
    model.compile(
        optimizer=Adam(1e-2),
        loss=composite_loss,
        metrics=['accuracy', iou_metric]
    )
    model.summary()
    print("Starting training...")

    callbacks = [
        ModelCheckpoint('best_flood_unet_tf.h5', save_best_only=True, monitor='val_loss', mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=8,
        callbacks=callbacks,
        verbose=1
    )

    # Plot training metrics
    epochs_rng = range(1, len(history.history['loss']) + 1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(epochs_rng, history.history['loss'], label='Train')
    plt.plot(epochs_rng, history.history['val_loss'], label='Val')
    plt.title('Loss'); plt.legend()

    plt.subplot(1,3,2)
    plt.plot(epochs_rng, history.history['accuracy'], label='Train')
    plt.plot(epochs_rng, history.history['val_accuracy'], label='Val')
    plt.title('Accuracy'); plt.legend()

    plt.subplot(1,3,3)
    plt.plot(epochs_rng, history.history['iou_metric'], label='Train')
    plt.plot(epochs_rng, history.history['val_iou_metric'], label='Val')
    plt.title('IoU'); plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics_tf.png')
    plt.close()

    # Visualize random predictions
    plt.figure(figsize=(12,8))
    for i in range(15):
        r = random.randint(0, N-1)
        img = X[r]
        pred = model.predict(img[np.newaxis,...])[0]
        mask_pred = (pred > 0.5).astype(np.uint8)
        plt.subplot(3,5,i+1)
        plt.imshow(img)
        plt.imshow(mask_pred.squeeze(), cmap='Blues', alpha=0.4)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('predictions_tf.png')
    plt.close()

    # Final evaluation
    model.load_weights('best_flood_unet_tf.h5')
    final = model.evaluate(X, Y, batch_size=8, verbose=1)
    print(f"Final - loss: {final[0]:.4f} - accuracy: {final[1]:.4f} - IoU: {final[2]:.4f}")
