import torch
import torch.nn as nn
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

class ViTClassifier:
    def __init__(self, model_path='models/vit_brain_tumor.pth', num_classes=4, class_names=None, dropout_rate=0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the pretrained model structure
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Replace classifier with the same Sequential used during training
        hidden_size = self.model.config.hidden_size
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Now load the saved state_dict (keys match because structure is identical)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()

        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        if class_names is None:
            self.class_names = ['glioma', 'meningioma', 'pituitary', 'notumor']
        else:
            self.class_names = class_names

    def predict(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_class_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_class_idx].item()
        return self.class_names[pred_class_idx], confidence, probs.cpu().numpy()