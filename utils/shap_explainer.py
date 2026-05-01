import shap
import torch
import numpy as np
import matplotlib.pyplot as plt

def explain_with_shap(model, image_tensor, class_names, background=None):
    model.eval()
    if background is None:
        background = torch.zeros_like(image_tensor)
    # GradientExplainer works with models that have gradients
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(image_tensor)
    # For image classification, shap.image_plot expects a list of images
    shap.image_plot(shap_values, image_tensor.cpu().numpy(), class_names=class_names)
    return shap_values