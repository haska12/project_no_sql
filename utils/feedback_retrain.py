import os
import shutil
from PIL import Image

def add_to_retrain_queue(image_path, true_label, predicted_label, confidence, queue_dir="data/retrain_queue"):
    class_dir = os.path.join(queue_dir, true_label)
    os.makedirs(class_dir, exist_ok=True)
    base = os.path.basename(image_path)
    new_name = f"{predicted_label}_{confidence:.2f}_{base}"
    dest = os.path.join(class_dir, new_name)
    shutil.copy(image_path, dest)
    return dest