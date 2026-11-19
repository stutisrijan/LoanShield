"""Train the optional signature CNN using ImageFolder directory layout.

Expected structure:
  data/signatures/train/genuine/*.png
  data/signatures/train/forged/*.png
  data/signatures/val/genuine/*.png
  data/signatures/val/forged/*.png

Use a properly licensed, labelled signature dataset. Do not use this model as
identity proof or as the sole basis for financial decisions.
"""
from pathlib import Path
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models.signature_cnn import SignatureCNN

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/signatures"
OUT = ROOT / "models/cnn/signature_cnn.pt"


def main(epochs=20, batch_size=32, learning_rate=1e-3):
    if not (DATA / "train").is_dir() or not (DATA / "val").is_dir():
        raise FileNotFoundError(
            "Signature dataset is missing. Create data/signatures/{train,val}/"
            "genuine and forged folders before training."
        )
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_ds = datasets.ImageFolder(DATA / "train", transform=tfm)
    val_ds = datasets.ImageFolder(DATA / "val", transform=tfm)
    if train_ds.classes != ["forged", "genuine"]:
        raise ValueError(f"Expected class folders forged/genuine; found {train_ds.classes}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignatureCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_accuracy = -1.0
    for epoch in range(epochs):
        model.train(); total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * len(labels)
        model.eval(); correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images.to(device))
                predicted = (torch.sigmoid(logits) >= 0.5).cpu().long()
                correct += int((predicted == labels).sum()); total += len(labels)
        accuracy = correct / max(total, 1)
        print(f"Epoch {epoch+1:02d}/{epochs} loss={total_loss/len(train_ds):.4f} val_acc={accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.cpu().state_dict(), OUT)
            model.to(device)
    (OUT.parent / "signature_metadata.json").write_text(json.dumps({
        "classes": train_ds.classes, "image_size": [128, 128],
        "validation_accuracy": best_accuracy, "device": str(device),
        "note": "Research prototype; not an identity verification guarantee."
    }, indent=2))
    print(f"Saved best signature CNN to {OUT}")


if __name__ == "__main__":
    main()
