import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from dataset import GliomaDataset
from models.unet import UNet


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dir = "data/images"
    mask_dir = "data/masks"

    dataset = GliomaDataset(img_dir, mask_dir)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    os.makedirs("results", exist_ok=True)

    with torch.no_grad():
        images, masks = next(iter(val_loader))
        images = images.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu()
        preds = (probs > 0.5).float()

    idx = 0
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(images[idx, 0].cpu(), cmap="gray")
    plt.title("Image")

    plt.subplot(1, 3, 2)
    plt.imshow(masks[idx, 0], cmap="gray")
    plt.title("Ground Truth")

    plt.subplot(1, 3, 3)
    plt.imshow(preds[idx, 0], cmap="gray")
    plt.title("Prediction")

    plt.tight_layout()
    plt.savefig("results/predictions.png")
    plt.show()


if __name__ == "__main__":
    evaluate()