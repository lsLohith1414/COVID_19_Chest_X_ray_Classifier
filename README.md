# COVID_19_Chest_X_ray_Classifier

**COVID_19_Chest_X_ray_Classifier** is a deep learning-based image classification system designed to automatically detect **COVID-19, Pneumonia, and Normal** cases from **chest X-ray images**. It leverages multiple pre-trained CNN models — **VGG19, ResNet50, and DenseNet121** — for feature extraction, enabling accurate and robust detection.

---

## Features

- **Multi-class Classification:** Detects **COVID-19 Positive**, **Pneumonia**, and **Normal** chest X-ray images.
- **Multiple Base Models:** Uses **VGG19, ResNet50, and DenseNet121** for transfer learning and feature extraction.
- **Preprocessing Pipeline:** Resizes, normalizes, and augments X-ray images for better training performance.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
- **Deployment Ready:** Can be deployed using **Flask or Streamlit** for real-time prediction.
- **Professional Folder Structure:** Organized folders for **data, src, models, artifacts, notebooks, app, config, and tests** for reproducibility.

---

## Project Structure

COVID_19_Chest_X_ray_Classifier/
├── data/ # Raw and processed X-ray images
├── notebooks/ # Jupyter notebooks for experimentation
├── src/ # Source code: preprocessing, training, evaluation
├── models/ # Saved trained models
├── artifacts/ # Training logs, plots, and evaluation reports
├── app/ # Flask/Streamlit deployment app
├── config/ # Config files (hyperparameters, paths)
├── tests/ # Unit tests for modules
├── requirements.txt # Python dependencies
└── README.md # Project description



---

## Tech Stack

- **Programming Language:** Python  
- **Deep Learning:** TensorFlow, Keras  
- **Pre-trained Models:** VGG19, ResNet50, DenseNet121  
- **Image Processing:** OpenCV, Pillow, NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment:** Flask / Streamlit  

---

## Use Case

- Supports **radiologists and healthcare professionals** in **early detection of COVID-19**.  
- Assists in **screening large volumes of chest X-ray images** automatically.  
- Provides a foundation for **further research in medical image analysis**.
