import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import ViTForImageClassification
from sklearn.metrics import accuracy_score, classification_report
import os
import gc
from utils.dataset import BrainTumorDataset
from torch.cuda.amp import autocast, GradScaler

def train_model_streamlit(data_dir, num_epochs, batch_size, lr, weight_decay, optimizer_name,
                          scheduler_step, dropout_rate, device, early_stop_accuracy, 
                          progress_callback=None, batch_callback=None, status_callback=None):
    
    # Clear memory before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    if status_callback:
        status_callback(f"Using device: {device}")

    full_dataset = BrainTumorDataset(data_dir)
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(model.config.hidden_size, num_classes)
    )
    model.to(device)

    if optimizer_name.lower() == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=scheduler_step)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    early_stopped = False

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(pixel_values=images)
                logits = outputs.logits
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
            if batch_callback:
                batch_callback(epoch+1, batch_idx+1, len(train_loader), loss.item())
                
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
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
        scheduler.step(avg_val_loss)

        if progress_callback:
            progress_callback(epoch+1, avg_train_loss, avg_val_loss, val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/vit_brain_tumor.pth')
            if status_callback:
                status_callback(f"Best model saved with acc={best_acc:.4f}")

        # --- Early stopping check ---
        if val_acc >= early_stop_accuracy:
            if status_callback:
                status_callback(f"Early stopping triggered! Validation accuracy {val_acc:.4f} reached target {early_stop_accuracy:.4f}.")
            early_stopped = True
            break

    report = classification_report(all_labels, all_preds, target_names=class_names)
    return model, history, report, class_names, early_stopped