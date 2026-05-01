import pydicom
from PIL import Image
import numpy as np
import io

def dicom_to_png(dicom_bytes):
    ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
    pixel_array = ds.pixel_array
    # Normalize to 0-255
    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8) * 255
    img = Image.fromarray(pixel_array.astype(np.uint8))
    metadata = {
        "PatientID": getattr(ds, 'PatientID', 'unknown'),
        "Modality": getattr(ds, 'Modality', 'unknown'),
        "StudyDate": getattr(ds, 'StudyDate', 'unknown'),
        "StudyDescription": getattr(ds, 'StudyDescription', '')
    }
    return img, metadata