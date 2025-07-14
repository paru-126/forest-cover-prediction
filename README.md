# 🌲 Forest Cover Type Prediction

This project uses machine learning to predict the forest cover type of a 30m x 30m patch of land in Roosevelt National Forest, Colorado, based on geographic and environmental features.

## 📌 Project Objective

The goal is to build a classification model that predicts the type of forest cover (e.g., Spruce/Fir, Lodgepole Pine, etc.) using tabular features such as elevation, slope, hillshade, soil type, wilderness area, etc.

---

## 📁 Dataset Description

The dataset was provided by the U.S. Forest Service and includes the following features:

- **Elevation**: Elevation in meters
- **Aspect**: Compass direction the slope faces
- **Slope**: Degree of incline
- **Hillshade (9am, Noon, 3pm)**: Amount of sunlight at different times of day
- **Horizontal & Vertical Distance** to water, fire points, and roads
- **Wilderness Area**: 4 binary columns representing protected areas
- **Soil Type**: 40 binary columns representing soil classification
- **Target Variable**:
  - 1 = Spruce/Fir
  - 2 = Lodgepole Pine
  - 3 = Ponderosa Pine
  - 4 = Cottonwood/Willow
  - 5 = Aspen
  - 6 = Douglas-fir
  - 7 = Krummholz

---

## 🧠 ML Workflow

1. **Data Loading**: Load the dataset with pandas
2. **Preprocessing**:
   - Scale numerical features
   - Split into train/test sets
3. **Model Training**:
   - Random Forest Classifier (baseline)
4. **Evaluation**:
   - Accuracy, Classification Report
   - Confusion Matrix
5. **Model Saving**:
   - Save trained model using `joblib`

---

## 📦 Project Structure

forest-cover-prediction/
│
├── train.csv # Dataset
├── forest_cover_prediction.py # Main script
├── model.pkl # Saved model after training
├── requirements.txt # Python package requirements
└── README.md # Project documentation

## To Run the Project
 -python forest_cover_prediction.py
