import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BrainTumorDataset(Dataset):
    """
    Custom Dataset for brain tumor MRI images.
    Handles:
    - Cleaning: filters only image files, skips corrupted/unreadable images.
    - Normalization: resizes to 224x224, converts to tensor, normalizes using ImageNet stats.
    - Missing values: corrupted images are detected and excluded during initialization.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: path to dataset root (e.g., 'data/train') containing class subfolders.
            transform: optional torchvision transforms (default = resize + totensor + normalize)
        """
        self.root_dir = root_dir
        # List all class folders (e.g., glioma, meningioma, ...)
        self.classes = sorted([d for d in os.listdir(root_dir) 
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Storage for valid image paths and labels
        self.image_paths = []
        self.labels = []
        
        # ---- CLEANING & HANDLING MISSING VALUES (corrupted files) ----
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                # 1. Filter only image files (cleaning)
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue  # skip non-image files
                
                img_path = os.path.join(cls_dir, img_name)
                # 2. Try to open the image to detect corruption (missing value handling)
                try:
                    with Image.open(img_path) as img:
                        img.verify()  # verify integrity
                    # If successful, add to list
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls])
                except (IOError, OSError, SyntaxError) as e:
                    # Corrupted or unreadable image -> skip (handle missing value)
                    print(f"Warning: Skipping corrupted image {img_path}: {e}")
                    # Optionally log to a file
        # ---- END OF CLEANING ----
        
        # ---- NORMALIZATION TRANSFORMS ----
        # Default transform: resize to 224x224 (ViT input size), convert to tensor,
        # and normalize using ImageNet mean/std (standard for transfer learning)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),          # scales pixels to [0,1]
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        # ---- END OF NORMALIZATION ----
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns a tuple (image_tensor, label) for the given index.
        """
        img_path = self.image_paths[idx]
        # Open image (already verified as valid during init)
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply normalization transforms
        if self.transform:
            image = self.transform(image)
        return image, label