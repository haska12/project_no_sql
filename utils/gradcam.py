import torch
import numpy as np
import cv2
from PIL import Image

class ViTGradCAM:
    def __init__(self, model):
        self.model = model
        self.attentions = None
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            # output is a tuple: (last_hidden_state, attentions)
            self.attentions = output[1]  # attentions from last layer
        # Register on the ViT model's encoder
        self.model.vit.encoder.register_forward_hook(hook_fn)

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        outputs = self.model(pixel_values=input_tensor, output_attentions=True)
        logits = outputs.logits
        target = logits[0, class_idx]
        target.backward()
        # attentions shape: (batch, layers, heads, seq_len, seq_len)
        attn = self.attentions[-1][0].mean(dim=0).detach().cpu().numpy()  # average over heads, last layer
        # Remove CLS token (first)
        attn = attn[1:, 1:]
        # Resize to 224x224
        size = int(np.sqrt(attn.shape[0]))
        attn_map = attn.mean(axis=0).reshape(size, size)
        attn_map = cv2.resize(attn_map, (224,224))
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        return attn_map