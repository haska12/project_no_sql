# Brain Tumor Classification with Vision Transformer (ViT) & MongoDB Dashboard

## Introduction

This project provides an end-to-end solution for **multi-class brain tumor classification** from MRI images using a **Vision Transformer (ViT)** fine-tuned on a public dataset. It includes a complete **Streamlit dashboard** that allows users to:

- Train the ViT model with custom hyperparameters.
- Monitor live training metrics (loss, accuracy, GPU/RAM usage).
- Save training runs and test evaluations to **MongoDB**.
- Predict single or batch MRI images.
- Generate diagnostic PDF reports.
- Analyse prediction statistics and model performance over time.

---

## Project Description

The project addresses the task of **classifying brain MRI scans** into four categories:

| Class        | Description         |
| :----------- | :------------------ |
| `glioma`     | Glioma tumour       |
| `meningioma` | Meningioma tumour   |
| `pituitary`  | Pituitary tumour    |
| `notumor`    | No tumour (healthy) |

**Key components:**

- **Vision Transformer (ViT):** Fine-tuned from `google/vit-base-patch16-224-in21k` using Hugging Face.
- **PyTorch:** Deep learning framework with mixed precision (AMP) training.
- **MongoDB:** Stores predictions, test evaluations, and training histories.
- **Streamlit:** Interactive dashboard for training, prediction, and analytics.

---

## Technologies Used

| Technology       | Purpose                    |
| :--------------- | :------------------------- |
| **Python 3.11**  | Core language              |
| **PyTorch**      | Deep learning framework    |
| **Hugging Face** | Pre-trained ViT model      |
| **Streamlit**    | Web dashboard              |
| **MongoDB**      | NoSQL database for metrics |
| **scikit-learn** | Evaluation metrics         |
| **FPDF**         | PDF report generation      |
| **pydicom**      | DICOM file support         |

---

## Functions

- **Training:** Fine-tune ViT with live loss/accuracy curves and hardware monitoring.
- **Prediction:** Single or batch upload (ZIP/DICOM). Returns class and confidence score.
- **Test Evaluation:** Generates confusion matrices, ROC curves, and F1 scores.
- **Analytics Dashboard:** Visualizes class distributions and historical training performance.
- **Retraining Queue:** Supports active learning by queuing images with corrected labels.

---

## Project Structure

```text
brain_tumor_vit/
├── data/                  # Train/Test datasets
├── models/                # Saved .pth files
├── utils/                 # Dataset, Inference, and Report logic
├── dashboard/             # app.py (Streamlit)
├── retrain.py             # Fine-tuning script
└── requirements.txt       # Dependencies
```

---

## Installation & Setup

1.  **Create Environment:**
    ```bash
    conda create -n brain_tumor_vit python=3.11 -y
    conda activate brain_tumor_vit
    ```
2.  **Install PyTorch (CUDA 12.1):**
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    ```
3.  **Install Dependencies:**
    ```bash
    pip install transformers streamlit pymongo pillow numpy matplotlib scikit-learn tqdm pandas seaborn psutil fpdf pydicom opencv-python shap
    ```
4.  **Run Dashboard:**
    ```bash
    streamlit run dashboard/app.py
    ```

---

### 💡 Quick Fix Tips:

- **Encoding:** If you save this as a file, ensure your text editor is set to **UTF-8** encoding.
- **Line Breaks:** Ensure you have an empty line between headers and paragraphs so the Markdown parser recognizes them correctly.
- **File Extension:** Save the file as `README.md`.

```

```
