import os
import torch
from torch.utils.data import DataLoader, random_split

from dataset import GliomaDataset
from models.unet import UNet
from utils import BCEDiceLoss, dice_score


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            dice = dice_score(outputs, masks)

            val_loss += loss.item()
            val_dice += dice.item()

    val_loss /= len(loader)
    val_dice /= len(loader)
    return val_loss, val_dice


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    img_dir = "data/images"
    mask_dir = "data/masks"

    dataset = GliomaDataset(img_dir, mask_dir)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = BCEDiceLoss()

    num_epochs = 30
    best_dice = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model.")

    print(f"Training complete. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    train()