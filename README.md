# Brain Tumor Classification with Vision Transformer (ViT) & MongoDB Dashboard

## Introduction

This project provides an end‑to‑end solution for **multi‑class brain tumor classification** from MRI images using a **Vision Transformer (ViT)** fine‑tuned on a public dataset. It includes a complete **Streamlit dashboard** that allows users to:

- Train the ViT model with custom hyperparameters.
- Monitor live training metrics (loss, accuracy, GPU/RAM usage).
- Save training runs and test evaluations to **MongoDB**.
- Predict single or batch MRI images.
- Generate diagnostic PDF reports.
- Analyse prediction statistics and model performance over time.

The system is designed for medical AI researchers, radiologists, and developers who want a production‑ready pipeline combining deep learning, NoSQL storage, and an interactive web interface.

---

## Project Description

The project addresses the task of **classifying brain MRI scans** into four categories:

| Class        | Description         |
| ------------ | ------------------- |
| `glioma`     | Glioma tumour       |
| `meningioma` | Meningioma tumour   |
| `pituitary`  | Pituitary tumour    |
| `notumor`    | No tumour (healthy) |

**Key components:**

- **Vision Transformer (ViT)** – fine‑tuned from `google/vit-base-patch16-224-in21k` using Hugging Face Transformers.
- **PyTorch** – deep learning framework with mixed precision (AMP) training.
- **MongoDB** – stores predictions, test evaluations, and training histories.
- **Streamlit** – interactive dashboard for training, prediction, analytics, and reporting.
- **Additional features**: Grad‑CAM (attention maps), SHAP explanations (optional), DICOM support, batch processing, PDF reports, early stopping, and export to CSV.

---

## Technologies Used

| Technology                    | Purpose                                                   |
| ----------------------------- | --------------------------------------------------------- |
| **Python 3.11**               | Core language                                             |
| **PyTorch**                   | Deep learning framework                                   |
| **Hugging Face Transformers** | Pre‑trained ViT model                                     |
| **Streamlit**                 | Web dashboard                                             |
| **MongoDB**                   | NoSQL database for predictions & metrics                  |
| **scikit‑learn**              | Evaluation metrics (accuracy, ROC, classification report) |
| **Matplotlib / Seaborn**      | Visualisations (loss curves, confusion matrix, ROC)       |
| **psutil / pynvml**           | System GPU/RAM monitoring                                 |
| **FPDF**                      | PDF report generation                                     |
| **pydicom**                   | DICOM file support                                        |
| **CUDA 12.1**                 | GPU acceleration (NVIDIA RTX 4050 tested)                 |

---

## Functions (What the System Does)

| Function                 | Description                                                                                                                                                                                                                                                            |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Training**             | Fine‑tune ViT on brain MRI dataset with configurable hyperparameters (batch size, learning rate, dropout, optimizer, early stopping). Displays live loss/accuracy curves, GPU utilisation, and RAM usage. Saves the best model and training history to MongoDB.        |
| **Prediction (Single)**  | Upload a single MRI image (JPEG, PNG, DICOM). Get tumour class and confidence score. Save prediction with optional true label and feedback to MongoDB.                                                                                                                 |
| **Batch Prediction**     | Upload a ZIP archive of images. All are processed in batch, saved to MongoDB, and results displayed with selection checkboxes. Export selected results as CSV or generate PDF reports for each.                                                                        |
| **Test Evaluation**      | Evaluate the trained model on a dedicated test folder. Computes accuracy, top‑2/3 accuracy, classification report, confusion matrix, and ROC curves. Stores all metrics and visualisations in MongoDB for later review.                                                |
| **Dashboard**            | Three tabs: **Test Evaluations** (browse saved test results, export PDF), **Predictions Analytics** (class distribution bar chart, confidence histogram, CSV export), **Training History** (view previous training runs with loss/accuracy plots and hyperparameters). |
| **PDF Reports**          | Generate a detailed diagnostic PDF for a patient, including name, age, gender, custom fields (text, date, time, URL, email, image), prediction, confidence, and the uploaded MRI image.                                                                                |
| **Retraining Queue**     | When a user corrects a prediction (provides true label), the image is added to a queue. A separate script (`retrain.py`) can fine‑tune the model on this feedback data (active learning).                                                                              |
| **DICOM Support**        | Automatically convert DICOM files to PNG and extract metadata (PatientID, Modality, StudyDate).                                                                                                                                                                        |
| **Model Explainability** | Placeholders for Grad‑CAM attention maps and SHAP values (can be extended).                                                                                                                                                                                            |

---

## How to Use the UI (Detailed Explanation)

The Streamlit dashboard is divided into **7 pages** (sidebar navigation). Below is a walkthrough of each page.

### 1. **Train & Evaluate**

- **Purpose**: Train a new ViT model from scratch (or fine‑tune).
- **Controls**:
  - Data directory (path to `train` folder with class subfolders).
  - Hyperparameters: epochs, batch size, learning rate, weight decay, optimizer (AdamW/SGD), scheduler patience, dropout rate.
  - Device: CUDA (GPU) or CPU.
  - **Early stopping accuracy threshold** – training stops automatically when validation accuracy reaches this value (default 0.98).
- **Live monitoring**:
  - Progress bar per epoch.
  - Loss/accuracy plots update after each epoch.
  - Status line shows current batch loss, GPU utilisation, and RAM usage.
- **After training**:
  - Best model saved to `models/vit_brain_tumor.pth`.
  - Training history (epochs, losses, accuracy) saved to MongoDB.
  - Download model button.

### 2. **Predict / Diagnose**

- **Upload** an MRI image (JPEG, PNG, or DICOM).
- Click **Diagnose** – returns predicted class and confidence.
- Optionally provide **true label** (for evaluation) and a comment.
- **Save to MongoDB** – stores prediction details.
- If true label differs from prediction, the image is added to the **retraining queue** (active learning).
- **Sidebar** shows the last 10 predictions.

### 3. **Test & Metrics**

- **Test dataset folder** – must have same subfolder structure as training (`glioma/`, `meningioma/`, `pituitary/`, `notumor/`).
- Enter an **evaluation name** (e.g., "baseline_v1").
- Click **Run Evaluation** – the model is evaluated on the entire test set.
- Results displayed:
  - Accuracy, Top‑2, Top‑3 accuracy.
  - Classification report (precision, recall, F1 per class).
  - Confusion matrix heatmap.
  - ROC curves (One‑vs‑Rest) with AUC values.
- All metrics and graphs are saved to MongoDB under the given evaluation name.

### 4. **Dashboard** (merged view)

Three tabs:

#### Test Evaluations

- Browse all saved test evaluations (newest first).
- Select one to display metrics, classification report, confusion matrix, and ROC curves.
- 95% confidence interval for accuracy (normal approximation).
- **Export Evaluation Report as PDF** – includes all metrics, report, and graphs.
- **Delete** button to remove an evaluation from MongoDB.

#### Predictions Analytics

- **Class distribution** bar chart (how many predictions per tumour type).
- **Confidence histogram** – shows how confident the model is across predictions.
- **Export all predictions as CSV** – downloads the entire `predictions` collection.

#### Training History

- Lists all previous training runs (timestamp, final validation accuracy).
- Select a run to view:
  - Configuration used (epochs, batch size, learning rate, etc.).
  - Loss/accuracy curves over epochs.
  - Final validation accuracy and classification report.

### 5. **Batch Predict**

- Upload a **ZIP file** containing multiple images (any folder structure).
- Click **Predict all** – processes all images, saves predictions to MongoDB.
- Results appear with **checkboxes** to select individual images.
- **Select All** / **Clear All** buttons.
- **Download selected as CSV** – exports only checked rows.
- **Generate PDF reports for selected** – creates a PDF report for each selected image (download links appear one by one).

### 6. **Reports** (single patient PDF)

- Enter **Patient Name** (mandatory), Age, Gender.
- **Add custom fields** (text, date, time, URL, email, image) – any number.
- Upload the MRI image.
- Click **Generate PDF Report** – downloads a PDF containing all patient data, prediction, confidence, and the MRI image.

### 7. **Retraining Queue**

- Shows the number of images waiting for fine‑tuning (added when a user corrects a prediction).
- **Start Fine‑tuning** button (runs a separate script that fine‑tunes the current model on the queued data).

---

## Project Structure

```
brain_tumor_vit/
├── data/
│   ├── train/                     # training images (glioma/, meningioma/, pituitary/, notumor/)
│   ├── test/                      # optional test set
│   └── retrain_queue/             # auto‑created for feedback images
├── models/                        # saved models (vit_brain_tumor.pth)
├── utils/
│   ├── __init__.py
│   ├── dataset.py                 # data loading, cleaning, normalization
│   ├── inference.py               # ViTClassifier class (prediction)
│   ├── train_utils.py             # training loop with callbacks
│   ├── roc_auc.py                 # ROC/AUC and top‑k metrics
│   ├── gradcam.py                 # Grad‑CAM attention maps (placeholder)
│   ├── shap_explainer.py          # SHAP explanations (placeholder)
│   ├── batch_predict.py           # batch prediction from ZIP
│   ├── report_generator.py        # PDF report generation
│   ├── feedback_retrain.py        # add to retraining queue
│   └── dicom_handler.py           # DICOM → PNG + metadata
├── dashboard/
│   └── app.py                     # main Streamlit application (all pages)
├── retrain.py                     # script to fine‑tune on queued images
├── requirements.txt
└── README.md
```

---

## Installation and Setup (Windows with NVIDIA RTX 4050)

### Prerequisites

- **Windows 10/11** with **NVIDIA GPU** (RTX 4050 or similar, 6 GB VRAM).
- **MongoDB** (Windows service) – install from [MongoDB Community Edition](https://www.mongodb.com/try/download/community) and start the service.
- **Python 3.11** (recommended, but 3.13 may work).
- **Anaconda** (optional but recommended for environment management).

### Step‑by‑Step Setup

1. **Clone or create the project folder**

   ```bash
   mkdir brain_tumor_vit
   cd brain_tumor_vit
   ```

2. **Create a Conda environment (Python 3.11)**

   ```bash
   conda create -n brain_tumor_vit python=3.11 -y
   conda activate brain_tumor_vit
   ```

3. **Install PyTorch with CUDA 12.1**

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install remaining dependencies**

   ```bash
   pip install transformers streamlit pymongo pillow numpy matplotlib scikit-learn tqdm pandas seaborn psutil fpdf pydicom opencv-python shap
   ```

5. **Start MongoDB (Windows service)**

   ```bash
   net start MongoDB
   ```

   Verify: `Test-NetConnection -ComputerName localhost -Port 27017` should succeed.

6. **Download the dataset**
   - Download [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle.
   - Extract and place the `Training/` folder inside `data/train` (i.e., `data/train/glioma/`, etc.).

7. **Run the Streamlit dashboard**
   ```bash
   streamlit run dashboard/app.py
   ```
   Open your browser at `http://localhost:8501`.

### Training your first model

- Go to **Train & Evaluate**.
- Set `Data directory = data/train`.
- Set `Batch size = 8` (or 16) – RTX 4050 6GB.
- Set `Device = cuda`.
- Keep `Early stop accuracy threshold = 0.98`.
- Click **Start Training**. Wait for completion (10‑15 minutes for 5‑10 epochs).
- The best model is saved automatically.

### Testing the model

- Place some test images in `data/test/` with the same subfolder structure.
- Go to **Test & Metrics**, enter `data/test`, give a name, and run evaluation.
- Results are saved to MongoDB and can be viewed in the Dashboard.

### Making predictions

- Go to **Predict / Diagnose**, upload an MRI image, click Diagnose.
- Save to MongoDB if desired.

---

## 📦 Requirements

Create a `requirements.txt` with the following (or install one by one as above):

```txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
streamlit>=1.25.0
pymongo>=4.5.0
pillow>=9.5.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
tqdm>=4.65.0
pandas>=1.5.0
seaborn>=0.12.0
psutil>=5.9.0
fpdf>=1.7.2
pydicom>=2.4.0
opencv-python>=4.8.0
shap>=0.41.0
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Troubleshooting

| Issue                                          | Solution                                                                                                  |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `CUDA out of memory`                           | Reduce batch size (e.g., 8). Enable mixed precision (already on).                                         |
| `ModuleNotFoundError: No module named 'utils'` | Run Streamlit from the project root, not inside `dashboard/`.                                             |
| MongoDB connection error                       | Ensure MongoDB service is running (`net start MongoDB`).                                                  |
| `torch.cuda.is_available() == False`           | Install correct PyTorch version with CUDA support. Verify driver version (`nvidia-smi`).                  |
| `bson.errors.InvalidDocument`                  | Already fixed in code – integer dictionary keys are converted to strings before inserting.                |
| Game performance drops after CUDA install      | This is unrelated to the project; reinstall NVIDIA Game Ready driver or use a separate Conda environment. |

---

## License

This project is for educational and research purposes. The dataset is from Kaggle under CC BY‑NC‑SA 4.0. The ViT model is from Hugging Face (Apache 2.0).

---

## Acknowledgements

- Hugging Face for the `transformers` library.
- Kaggle user **masoudnickparvar** for the brain tumor MRI dataset.
- Streamlit for the amazing dashboard framework.

---

**Enjoy diagnosing brain tumours with AI!**
