import streamlit as st
import pymongo
from PIL import Image
import datetime
import sys
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import zipfile
import tempfile
import psutil
import subprocess
import gc
import io
import base64
import math
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from fpdf import FPDF

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.inference import ViTClassifier
from utils.train_utils import train_model_streamlit
from utils.dataset import BrainTumorDataset
from utils.roc_auc import plot_multiclass_roc, compute_topk_accuracy
from utils.batch_predict import batch_predict_from_zip
from utils.report_generator import generate_pdf_report
from utils.feedback_retrain import add_to_retrain_queue
from utils.dicom_handler import dicom_to_png

# ---------- GPU & RAM Monitoring ----------
def get_gpu_stats():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            util, mem_used = result.stdout.strip().split(', ')
            return int(util), int(mem_used)
    except:
        pass
    return 0, 0

def get_ram_usage():
    return psutil.virtual_memory().percent

# ---------- MongoDB ----------
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "brain_tumor_db"
COLLECTION_NAME = "predictions"
EVAL_COLLECTION_NAME = "test_evaluations"
TRAINING_COLLECTION_NAME = "training_runs"

@st.cache_resource
def init_mongo():
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    eval_collection = db[EVAL_COLLECTION_NAME]
    training_collection = db[TRAINING_COLLECTION_NAME]
    collection.create_index([("timestamp", pymongo.DESCENDING)])
    eval_collection.create_index([("timestamp", pymongo.DESCENDING)])
    training_collection.create_index([("timestamp", pymongo.DESCENDING)])
    return collection, eval_collection, training_collection

st.set_page_config(page_title="Brain Tumor ViT Suite", layout="wide")

dark_mode = st.sidebar.checkbox("Dark mode")
if dark_mode:
    st._config.set_option("theme.base", "dark")
else:
    st._config.set_option("theme.base", "light")

collection, eval_collection, training_collection = init_mongo()

# Navigation
page = st.sidebar.radio("Navigation", 
    ["Train and Evaluate", "Predict / Diagnose", "Test and Metrics", 
     "Dashboard", "Batch Predict", "Reports", "Retraining Queue"])

# ===================================================
# PAGE 1: TRAINING
# ===================================================
if page == "Train and Evaluate":
    st.title("Train Vision Transformer")
    
    with st.form("training_form"):
        col1, col2 = st.columns(2)
        with col1:
            data_dir = st.text_input("Data directory", "data/train")
            num_epochs = st.number_input("Epochs", 1, 50, 10)
            batch_size = st.number_input("Batch size", 4, 128, 8)
            lr = st.number_input("Learning rate", 1e-6, 1e-3, 2e-5, format="%.6f")
        with col2:
            weight_decay = st.number_input("Weight decay", 0.0, 0.1, 0.01, format="%.4f")
            optimizer_name = st.selectbox("Optimizer", ["AdamW", "SGD"])
            scheduler_step = st.number_input("Scheduler patience", 1, 10, 2)
            dropout_rate = st.slider("Dropout", 0.0, 0.5, 0.1)
            device = st.selectbox("Device", ["cuda", "cpu"], index=0 if torch.cuda.is_available() else 1)
            early_stop_accuracy = st.slider("Early stop accuracy threshold", 0.80, 0.99, 0.98, 0.01,
                                            help="Stop training when validation accuracy reaches this value")
        submitted = st.form_submit_button("Start Training")
    
    if submitted:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        status = st.empty()
        progress_bar = st.progress(0)
        metrics_col1 = st.columns(1)[0]
        metrics_col1.metric("Validation Accuracy", "0.00")
        plot_placeholder = st.empty()
        epoch_history = []
        
        def status_cb(msg): status.info(msg)
        
        def prog_cb(epoch, train_loss, val_loss, val_acc):
            progress_bar.progress(epoch / num_epochs)
            metrics_col1.metric("Validation Accuracy", f"{val_acc:.4f}")
            epoch_history.append((epoch, train_loss, val_loss, val_acc))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            e = [h[0] for h in epoch_history]
            ax1.plot(e, [h[1] for h in epoch_history], label='Train Loss')
            ax1.plot(e, [h[2] for h in epoch_history], label='Val Loss')
            ax1.legend()
            ax2.plot(e, [h[3] for h in epoch_history], label='Val Acc', color='green')
            ax2.legend()
            plot_placeholder.pyplot(fig)
            plt.close(fig)
        
        def batch_cb(epoch, batch_idx, total_batches, loss):
            gpu_util, _ = get_gpu_stats()
            ram_percent = get_ram_usage()
            status_cb(f"Epoch {epoch}, Batch {batch_idx}/{total_batches} | Loss: {loss:.4f} | GPU: {gpu_util}% | RAM: {ram_percent}%")
        
        model, hist, report, class_names, early_stopped = train_model_streamlit(
            data_dir, num_epochs, batch_size, lr, weight_decay, optimizer_name,
            scheduler_step, dropout_rate, device, early_stop_accuracy,
            prog_cb, batch_cb, status_cb
        )
        
        if early_stopped:
            st.info(f"Training stopped early because validation accuracy reached {early_stop_accuracy:.2%}.")
        
        if epoch_history:
            epochs_list = [{"epoch": e[0], "train_loss": e[1], "val_loss": e[2], "val_accuracy": e[3]} for e in epoch_history]
            final_acc = epoch_history[-1][3] if epoch_history else 0.0
            training_doc = {
                "timestamp": datetime.datetime.utcnow(),
                "config": {
                    "data_dir": data_dir,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                    "optimizer": optimizer_name,
                    "dropout_rate": dropout_rate,
                    "early_stop_accuracy": early_stop_accuracy
                },
                "epochs": epochs_list,
                "final_validation_accuracy": final_acc,
                "classification_report": report,
                "early_stopped": early_stopped
            }
            training_collection.insert_one(training_doc)
            st.success("Training metrics saved to MongoDB.")
        
        st.success("Training completed! Model saved to models/vit_brain_tumor.pth")
        with open("models/vit_brain_tumor.pth", "rb") as f:
            st.download_button("Download model", f, "vit_brain_tumor.pth")

# ===================================================
# PAGE 2: PREDICT / DIAGNOSE
# ===================================================
elif page == "Predict / Diagnose":
    st.title("Predict Single Image")
    if not os.path.exists('models/vit_brain_tumor.pth'):
        st.error("No model found. Train first.")
        st.stop()
    model = ViTClassifier()
    st.markdown("""
    <style>
    div[data-testid="stFileUploader"] { width: 100%; padding: 30px; border: 2px dashed #ccc; border-radius: 10px; background-color: #f9f9f9; }
    div[data-testid="stFileUploader"]:hover { border-color: #007bff; }
    </style>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag and drop or click", type=["jpg","jpeg","png","dcm"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.dcm'):
            img, metadata = dicom_to_png(uploaded_file.getvalue())
            st.write("DICOM metadata:", metadata)
        else:
            img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Input image", width=300)
        if st.button("Diagnose"):
            pred, conf, probs = model.predict(img)
            st.success(f"**Prediction: {pred}** (confidence: {conf:.2%})")
            if st.checkbox("Show attention map (Grad-CAM)"):
                st.info("Grad-CAM would appear here (needs model wrapper).")
            true_label = st.selectbox("True label (optional)", model.class_names, index=model.class_names.index(pred))
            feedback = st.text_input("Comment")
            if st.button("Save to MongoDB"):
                doc = {"filename": uploaded_file.name, "timestamp": datetime.datetime.utcnow(),
                       "true_label": true_label, "predicted_label": pred, "confidence": conf, "user_feedback": feedback}
                collection.insert_one(doc)
                if true_label != pred:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    add_to_retrain_queue(temp_path, true_label, pred, conf)
                    os.remove(temp_path)
                    st.info("Added to retraining queue.")
                st.success("Saved!")
                st.rerun()
    with st.sidebar:
        st.subheader("Recent predictions")
        recent = list(collection.find().sort("timestamp", -1).limit(10))
        for r in recent:
            st.write(f"{r['filename']} -> {r['predicted_label']} ({r['confidence']:.2f})")

# ===================================================
# PAGE 3: TEST & METRICS
# ===================================================
elif page == "Test and Metrics":
    st.title("Evaluate on Test Set")
    test_dir = st.text_input("Test dataset folder", "data/test")
    eval_name = st.text_input("Evaluation name", "test_run_1")
    
    if st.button("Run Evaluation"):
        if not os.path.exists(test_dir):
            st.error("Folder not found")
        else:
            with st.spinner("Running evaluation on test set... This may take a few moments."):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = ViTClassifier()
                model.model.to(device)
                test_dataset = BrainTumorDataset(test_dir)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
                all_preds, all_labels, all_probs = [], [], []
                for images, labels in test_loader:
                    images = images.to(device)
                    with torch.no_grad():
                        outputs = model.model(pixel_values=images)
                        logits = outputs.logits
                        probs = torch.nn.functional.softmax(logits, dim=1)
                        preds = torch.argmax(probs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())
                    all_probs.extend(probs.cpu().numpy())
                all_labels = np.array(all_labels)
                all_probs = np.array(all_probs)
                
                acc = accuracy_score(all_labels, all_preds)
                top2 = compute_topk_accuracy(all_labels, all_probs, k=2)
                top3 = compute_topk_accuracy(all_labels, all_probs, k=3)
                report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, output_dict=True)
                cm = confusion_matrix(all_labels, all_preds)
                roc_fig, roc_auc = plot_multiclass_roc(all_labels, all_probs, test_dataset.classes)
                
                def fig_to_base64(fig):
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    buf.seek(0)
                    return base64.b64encode(buf.read()).decode('utf-8')
                
                cm_fig, ax = plt.subplots(figsize=(5,4))
                sns.heatmap(cm, annot=True, fmt='d', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes, ax=ax)
                ax.set_title("Confusion Matrix")
                cm_base64 = fig_to_base64(cm_fig)
                plt.close(cm_fig)
                
                roc_base64 = fig_to_base64(roc_fig)
                plt.close(roc_fig)
                
                def dict_keys_to_str(d):
                    new = {}
                    for k, v in d.items():
                        if isinstance(k, (int, np.integer)):
                            new[str(k)] = v
                        else:
                            new[k] = v
                    return new
                
                report_str = dict_keys_to_str(report)
                roc_auc_str = dict_keys_to_str(roc_auc)
                
                eval_doc = {
                    "eval_name": eval_name,
                    "timestamp": datetime.datetime.utcnow(),
                    "test_dir": test_dir,
                    "accuracy": acc,
                    "top2_accuracy": top2,
                    "top3_accuracy": top3,
                    "classification_report": report_str,
                    "confusion_matrix_base64": cm_base64,
                    "roc_curve_base64": roc_base64,
                    "roc_auc": roc_auc_str,
                    "num_samples": len(test_dataset)
                }
                eval_collection.insert_one(eval_doc)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("Top-2 Accuracy", f"{top2:.4f}")
                col3.metric("Top-3 Accuracy", f"{top3:.4f}")
                
                st.dataframe(pd.DataFrame(report).transpose(), use_container_width=False, height=280)
                col_left, col_right = st.columns(2)
                col_left.image(io.BytesIO(base64.b64decode(cm_base64)), caption="Confusion Matrix")
                col_right.image(io.BytesIO(base64.b64decode(roc_base64)), caption="ROC Curves")
                st.success(f"Evaluation saved to MongoDB with name: {eval_name}")

# ===================================================
# PAGE 4: DASHBOARD
# ===================================================
elif page == "Dashboard":
    st.title("Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["Test Evaluations", "Predictions Analytics", "Training History"])
    
    with tab1:
        st.subheader("Saved Test Evaluation Results")
        evaluations = list(eval_collection.find().sort("timestamp", -1))
        if not evaluations:
            st.info("No test evaluations found. Run a test evaluation first.")
        else:
            eval_names = [e["eval_name"] + f" ({e['timestamp'].strftime('%Y-%m-%d %H:%M')})" for e in evaluations]
            selected_idx = st.selectbox("Select evaluation to display", range(len(eval_names)), format_func=lambda i: eval_names[i], key="eval_selector")
            selected_eval = evaluations[selected_idx]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{selected_eval['accuracy']:.4f}")
            col2.metric("Top-2 Accuracy", f"{selected_eval['top2_accuracy']:.4f}")
            col3.metric("Top-3 Accuracy", f"{selected_eval['top3_accuracy']:.4f}")
            col4.metric("Test Samples", selected_eval['num_samples'])
            
            acc = selected_eval['accuracy']
            n = selected_eval['num_samples']
            se = math.sqrt(acc*(1-acc)/n) if n > 0 else 0
            ci_lower = acc - 1.96*se
            ci_upper = acc + 1.96*se
            st.caption(f"95% Confidence Interval for Accuracy: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            st.subheader("Classification Report")
            report_df = pd.DataFrame(selected_eval['classification_report']).transpose()
            st.dataframe(report_df, use_container_width=True, height=300)
            
            st.subheader("Visualizations")
            col_img1, col_img2 = st.columns(2)
            col_img1.image(io.BytesIO(base64.b64decode(selected_eval['confusion_matrix_base64'])), caption="Confusion Matrix", use_column_width=True)
            col_img2.image(io.BytesIO(base64.b64decode(selected_eval['roc_curve_base64'])), caption="ROC Curves", use_column_width=True)
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Export Evaluation Report as PDF"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(200, 10, txt="Brain Tumor ViT - Test Evaluation Report", ln=1, align="C")
                    pdf.ln(10)
                    pdf.set_font("Arial", "", 12)
                    pdf.cell(200, 8, txt=f"Evaluation: {selected_eval['eval_name']}", ln=1)
                    pdf_output = pdf.output(dest='S').encode('latin-1')
                    st.download_button("Download PDF", pdf_output, f"{selected_eval['eval_name']}_report.pdf", "application/pdf")
            with col_btn2:
                if st.button("Delete this evaluation"):
                    eval_collection.delete_one({"_id": selected_eval["_id"]})
                    st.rerun()
    
    with tab2:
        st.subheader("Predictions Analytics")
        pipeline_class = [{"$group": {"_id": "$predicted_label", "count": {"$sum": 1}}}]
        class_stats = list(collection.aggregate(pipeline_class))
        data = list(collection.find({}, {"confidence": 1}))
        
        if class_stats or data:
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Predicted Class Distribution**")
                if class_stats:
                    df_class = pd.DataFrame(class_stats)
                    df_class.set_index("_id", inplace=True)
                    st.bar_chart(df_class)
            with col_right:
                st.markdown("**Confidence Distribution**")
                if data:
                    conf_vals = [d['confidence'] for d in data]
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.hist(conf_vals, bins=20, edgecolor='black', color='skyblue')
                    st.pyplot(fig)
                    plt.close(fig)
        
        if st.button("Export all predictions as CSV"):
            cursor = collection.find({}, {"_id": 0})
            df = pd.DataFrame(list(cursor))
            if not df.empty:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "predictions.csv", "text/csv")

    with tab3:
        st.subheader("Training History")
        training_runs = list(training_collection.find().sort("timestamp", -1))
        if not training_runs:
            st.info("No training runs found.")
        else:
            run_names = [f"Run {i+1}: {run['timestamp'].strftime('%Y-%m-%d %H:%M')}" for i, run in enumerate(training_runs)]
            selected_run_idx = st.selectbox("Select training run to view", range(len(run_names)), format_func=lambda i: run_names[i], key="training_selector")
            selected_run = training_runs[selected_run_idx]
            
            st.subheader("Configuration")
            config = selected_run['config']
            st.json(config)
            
            epochs = [e['epoch'] for e in selected_run['epochs']]
            train_loss = [e['train_loss'] for e in selected_run['epochs']]
            val_loss = [e['val_loss'] for e in selected_run['epochs']]
            val_acc = [e['val_accuracy'] for e in selected_run['epochs']]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(epochs, train_loss, label='Train Loss')
            ax1.plot(epochs, val_loss, label='Val Loss')
            ax1.legend()
            ax2.plot(epochs, val_acc, label='Val Accuracy', color='green')
            ax2.legend()
            st.pyplot(fig)
            plt.close(fig)

# ===================================================
# PAGE 5: BATCH PREDICTION
# ===================================================
elif page == "Batch Predict":
    st.title("Batch Prediction from ZIP")
    zip_file = st.file_uploader("Upload ZIP file with images", type=["zip"])
    if zip_file and st.button("Predict all"):
        with st.spinner("Processing ZIP file and running predictions..."):
            model = ViTClassifier()
            results = batch_predict_from_zip(zip_file.getvalue(), model)
        if results:
            st.session_state['batch_results'] = results
            st.session_state['batch_zip_bytes'] = zip_file.getvalue()
            st.session_state['batch_selected'] = [False] * len(results)
            st.success(f"Prediction completed for {len(results)} images.")
            for fname, pred, conf in results:
                doc = {"filename": fname, "timestamp": datetime.datetime.utcnow(),
                       "true_label": "unknown", "predicted_label": pred, "confidence": conf,
                       "user_feedback": "batch prediction", "batch": True}
                collection.insert_one(doc)
    
    if 'batch_results' in st.session_state and st.session_state['batch_results']:
        results = st.session_state['batch_results']
        st.subheader("Prediction Results - Select images")
        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("Select All"):
            st.session_state['batch_selected'] = [True] * len(results)
            st.rerun()
        if col_btn2.button("Clear All"):
            st.session_state['batch_selected'] = [False] * len(results)
            st.rerun()
        for idx, (fname, pred, conf) in enumerate(results):
            col1, col2, col3, col4 = st.columns([0.5, 3, 2, 2])
            selected = col1.checkbox("", value=st.session_state['batch_selected'][idx], key=f"batch_sel_{idx}")
            st.session_state['batch_selected'][idx] = selected
            col2.write(f"**{fname}**")
            col3.write(f"Prediction: {pred}")
            col4.write(f"Confidence: {conf:.2%}")

# ===================================================
# PAGE 6: REPORTS
# ===================================================
elif page == "Reports":
    st.title("Generate PDF Report")
    if "extra_fields" not in st.session_state:
        st.session_state.extra_fields = []
    col1, col2, col3 = st.columns(3)
    patient_name = col1.text_input("Patient Name *", key="patient_name")
    patient_age = col2.text_input("Age", key="patient_age")
    patient_gender = col3.selectbox("Gender", ["", "Male", "Female", "Other"], key="patient_gender")