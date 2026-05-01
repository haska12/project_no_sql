import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import os

from dataset import BrainTumorDataset

def train_model(data_dir='data/train', epochs=10, batch_size=16, lr=2e-5, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    full_dataset = BrainTumorDataset(data_dir)
    num_classes = len(full_dataset.classes)
    print(f"Classes: {full_dataset.classes}")
    print(f"Total images: {len(full_dataset)}")

    # Train/Val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load ViT
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images)
            logits = outputs.logits
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(pixel_values=images)
                logits = outputs.logits
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/vit_brain_tumor.pth')
            print(f"Best model saved with accuracy {best_acc:.4f}")

    # Final report
    print("\nClassification report on validation set:")
    print(classification_report(all_labels, all_preds, target_names=full_dataset.classes))

    # Save plots
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss curves')
    plt.savefig('loss_curves.png')

    plt.figure()
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig('accuracy.png')

    return model

if __name__ == '__main__':
    train_model()