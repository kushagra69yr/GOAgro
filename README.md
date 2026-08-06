# 🌾 GOAgro – AI-Powered Smart Agriculture Platform

GOAgro is an AI-powered smart farming platform that leverages Machine Learning to recommend the most suitable crop based on soil nutrients and environmental conditions. Built using Flask, Python, and Scikit-Learn, the platform enables farmers to make data-driven agricultural decisions through an intuitive and responsive web interface.

## 🚀 Features

- 🌱 AI-based Crop Recommendation using Random Forest Classifier
- 📊 99.1% Model Accuracy
- 🌦️ Prediction based on Soil Nutrients & Weather Parameters
- 💧 Water Requirement & Fertilizer Recommendations
- 🌾 Crop Yield & MSP Information
- 📖 Detailed Cultivation Guidelines
- 📱 Modern Responsive Glassmorphism UI
- ⚡ Fast Real-time Predictions

## 🏗️ System Architecture

```text
User
   │
   ▼
Frontend (HTML/CSS/JavaScript)
   │
   ▼
Flask Backend
   │
   ▼
Input Preprocessing (NumPy)
   │
   ▼
Random Forest Model (.pkl)
   │
   ▼
Crop Prediction
   │
   ▼
Crop Details & Farming Guide
```

## 🤖 Machine Learning

- **Algorithm:** Random Forest Classifier
- **Dataset:** Crop Recommendation Dataset
- **Features Used:**
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature
  - Humidity
  - pH
  - Rainfall
- **Validation Accuracy:** 99.1%

### Why Random Forest?

- High accuracy on structured agricultural datasets
- Reduces overfitting through ensemble learning
- Fast and reliable predictions
- Handles nonlinear relationships effectively

## 🛠️ Tech Stack

**Frontend**
- HTML5
- CSS3
- JavaScript

**Backend**
- Flask
- Python

**Machine Learning**
- Scikit-Learn
- NumPy
- Pandas
- Joblib

## ⚙️ Workflow

1. User enters soil and weather parameters.
2. Flask validates and preprocesses the input.
3. Data is converted into a NumPy array.
4. The trained Random Forest model predicts the best crop.
5. The application displays cultivation details, fertilizer recommendations, water requirements, expected yield, and MSP.

## 🚀 Future Enhancements

- 🌿 Leaf Disease Detection using Deep Learning
- ☁️ Live Weather API Integration
- 📍 GPS-based Crop Recommendation
- 🛒 Farmer Marketplace
- 🌍 Multi-language Support
- 📱 Android & iOS Mobile Application
- 🌡️ IoT Soil Sensor Integration

## 💡 Project Objective

The goal of GOAgro is to assist farmers in selecting the most suitable crop using Artificial Intelligence, improving agricultural productivity while reducing dependency on traditional trial-and-error methods.

⭐ If you found this project useful, consider giving it a Star!
