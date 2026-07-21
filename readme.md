# 🏃 Human Activity Recognition (HAR) Streamlit App

A clean, interactive Streamlit web application that predicts human physical activities (such as *Walking, Running, Sitting, Standing, and Lying*) from mobile accelerometer and gyroscope sensor data using a trained Machine Learning model.

---

## ✨ Features

- 📁 **CSV File Upload & Validation:** Upload custom sensor data in CSV format with automatic validation against expected feature columns.
- ⬇️ **Sample Template Download:** Download a pre-formatted template CSV to easily align user datasets.
- 🧪 **Sample Data Mode:** Quick testing option using built-in sample or synthetic dataset inputs.
- 🔮 **Real-Time Prediction:** Instant activity predictions along with visual confidence bars and class probabilities.
- 📊 **Interactive Visualizations:** View probability distribution bar charts for predicted activity categories.
- 📥 **Batch Prediction Export:** Download the input dataset attached with predicted labels and confidence scores in CSV format.

---

## 🛠️ Tech Stack

- **Frontend / UI:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning Integration:** Scikit-Learn / Joblib
- **Language:** Python 3.x

---

## 📁 Project Structure

```text
├── app.py                   # Main Streamlit application
├── stacked_model.pkl        # Trained Machine Learning model file
├── expected_columns.txt     # List of required sensor feature names
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation
