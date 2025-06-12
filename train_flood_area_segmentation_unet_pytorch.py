import os
import random
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from tqdm.auto import tqdm

# -------- Dataset Definition --------
class FloodDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_transform=None, mask_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()
        return img, mask

# -------- U-Net Model --------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[8,16,32,64,128,256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)
        # Down path
        for feat in features:
            self.downs.append(DoubleConv(in_channels, feat))
            in_channels = feat
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        prev_channels = features[-1]*2
        # Up path
        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose2d(prev_channels, feat, 2,2))
            self.ups.append(DoubleConv(prev_channels, feat))
            prev_channels = feat
        self.final_conv = nn.Conv2d(prev_channels, out_channels, 1)

    def forward(self, x):
        skip = []
        for down in self.downs:
            x = down(x)
            skip.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip = skip[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            s = skip[idx//2]
            if x.shape != s.shape:
                x = nn.functional.interpolate(x, size=s.shape[2:])
            x = torch.cat([s, x], dim=1)
            x = self.ups[idx+1](x)
        return torch.sigmoid(self.final_conv(x))

# -------- Losses & Metrics --------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, preds, targets):
        preds_flat = preds.view(preds.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (preds_flat * targets_flat).sum(dim=1)
        dice = (2 * intersection + self.eps) / (preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.eps)
        return 1 - dice.mean()

def iou_metric(preds, targets, threshold=0.3, eps=1e-6):
    preds = (preds > threshold).float()
    inter = (preds * targets).sum((1,2,3))
    union = (preds + targets - preds * targets).sum((1,2,3))
    return ((inter + eps) / (union + eps)).mean().item()

# -------- Train/Val Epochs --------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = acc_sum = iou_sum = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        acc_sum += ((preds > 0.5) == masks).float().mean().item()
        iou_sum += iou_metric(preds, masks)
    n = len(loader)
    return loss_sum/n, acc_sum/n, iou_sum/n

def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum = acc_sum = iou_sum = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss_sum += loss.item()
            acc_sum += ((preds > 0.5) == masks).float().mean().item()
            iou_sum += iou_metric(preds, masks)
    n = len(loader)
    return loss_sum/n, acc_sum/n, iou_sum/n

# -------- Main --------
def main():
    base = Path(__file__).parent
    img_paths = sorted((base/'Image').glob('*.jpg'))
    msk_paths = sorted((base/'Mask').glob('*.png'))
    assert len(img_paths) == len(msk_paths), "Mismatch"

    # Separate transforms for images and masks
    img_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256,256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    dataset = FloodDataset(img_paths, msk_paths, img_transform, mask_transform)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=40, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=40)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    bce_loss = nn.BCELoss()
    dice_loss = DiceLoss()
    criterion = lambda preds, masks: bce_loss(preds, masks) + dice_loss(preds, masks)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    epochs = 100
    best_loss = float('inf')
    patience = 0
    early_stop_patience = 10
    history = {'tr_loss':[], 'tr_acc':[], 'tr_iou':[], 'vl_loss':[], 'vl_acc':[], 'vl_iou':[]}

    for e in range(1, epochs+1):
        start = time.time()
        tr_iter = tqdm(train_loader, desc=f"Epoch {e}/{epochs} [Train]", leave=False)
        tr_loss, tr_acc, tr_iou = train_epoch(model, tr_iter, criterion, optimizer, device)
        vl_iter = tqdm(val_loader, desc=f"Epoch {e}/{epochs} [Val]  ", leave=False)
        vl_loss, vl_acc, vl_iou = eval_epoch(model, vl_iter, criterion, device)

        history['tr_loss'].append(tr_loss)
        history['tr_acc'].append(tr_acc)
        history['tr_iou'].append(tr_iou)
        history['vl_loss'].append(vl_loss)
        history['vl_acc'].append(vl_acc)
        history['vl_iou'].append(vl_iou)

        # Check improvement
        if vl_loss < best_loss:
            print(f"Epoch {e}: val_loss improved from {best_loss:.5f} to {vl_loss:.5f}, saving model to best_flood_unet.pth")
            best_loss = vl_loss
            patience = 0
            torch.save(model.state_dict(), 'best_flood_unet.pth')
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping triggered at epoch {e} (patience={patience})")
                break

        scheduler.step(vl_loss)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start
        print(f"{e}/{epochs} - {elapsed:.1f}s - loss: {tr_loss:.4f} - acc: {tr_acc:.4f} - iou: {tr_iou:.4f} - "
              f"vl_loss: {vl_loss:.4f} - vl_acc: {vl_acc:.4f} - vl_iou: {vl_iou:.4f} - lr: {lr:.1e}")

    # Save training metrics plot
    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1); plt.plot(history['tr_loss'],label='Train'); plt.plot(history['vl_loss'],label='Val'); plt.title('Loss'); plt.legend()
    plt.subplot(1,3,2); plt.plot(history['tr_acc'],label='Train'); plt.plot(history['vl_acc'],label='Val'); plt.title('Accuracy'); plt.legend()
    plt.subplot(1,3,3); plt.plot(history['tr_iou'],label='Train'); plt.plot(history['vl_iou'],label='Val'); plt.title('IoU'); plt.legend()
    plt.tight_layout(); plt.savefig('training_metrics_pytorch.png'); plt.close()

    # Visualize predictions
    plt.figure(figsize=(12,8))
    for i in range(15):
        idx = random.randint(0, len(dataset)-1)
        img, _ = dataset[idx]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device)) > 0.5
        pred = pred.cpu().squeeze().numpy()
        plt.subplot(3,5,i+1)
        plt.imshow(transforms.ToPILImage()(img))
        plt.imshow(pred, cmap='Blues', alpha=0.4)
        plt.axis('off')
    plt.tight_layout(); plt.savefig('predictions_pytorch.png'); plt.close()

    # Final evaluation on full dataset
    model.load_state_dict(torch.load('best_flood_unet.pth', map_location=device))
    final_loss, final_acc, final_iou = eval_epoch(model, DataLoader(dataset, batch_size=40), criterion, device)
    print(f"Final - loss: {final_loss:.4f} - acc: {final_acc:.4f} - iou: {final_iou:.4f}")

if __name__ == '__main__':
    main()
