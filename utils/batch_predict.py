import zipfile
import tempfile
import os
from PIL import Image
from utils.inference import ViTClassifier

def batch_predict_from_zip(zip_bytes, model):
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmpdir)
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file.lower().endswith(('.png','.jpg','.jpeg')):
                        img_path = os.path.join(root, file)
                        img = Image.open(img_path).convert('RGB')
                        pred, conf, _ = model.predict(img)
                        # Use relative path from zip root as filename
                        rel_path = os.path.relpath(img_path, tmpdir)
                        results.append((rel_path, pred, conf))
    return results