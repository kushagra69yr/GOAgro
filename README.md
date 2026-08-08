# 🌾 GOAgro

## AI-Powered Smart Agriculture Platform

GOAgro is a machine-learning-based agriculture platform that uses **soil nutrients and environmental conditions** to recommend a suitable crop. The project combines a Flask web application with a trained **Random Forest Classifier** and crop-specific farming information to turn structured field data into an accessible decision-support workflow.

> **Agricultural Inputs → Machine Learning → Crop Recommendation → Farming Guidance**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/) [![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) [![Status](https://img.shields.io/badge/Status-Active-2ea44f)](https://github.com/kushagra69yr/GOAgro)

## 🚀 What the Project Does

Farmers make crop decisions using several variables at once. GOAgro brings those variables into one interface and applies a trained machine-learning model to recommend a crop.

The current prediction pipeline uses seven inputs:

| Input | Role |
|---|---|
| **Nitrogen (N)** | Soil nutrient level |
| **Phosphorus (P)** | Soil nutrient level |
| **Potassium (K)** | Soil nutrient level |
| **Temperature** | Environmental condition |
| **Humidity** | Environmental condition |
| **pH** | Soil acidity / alkalinity |
| **Rainfall** | Water availability indicator |

The predicted crop can then be presented with the farming information available in the application, including cultivation guidance and other crop details.

## ✨ Key Features

- 🌱 **Crop recommendation** using a Random Forest Classifier
- 🧪 Seven soil and environmental input features
- ⚡ Flask-based real-time inference workflow
- 📊 Reported **99.1% validation accuracy** for the model
- 💧 Water requirement information
- 🌿 Fertilizer recommendations
- 🌾 Crop yield information
- 💰 MSP / market information where provided by the application
- 📖 Crop-specific cultivation guidance
- 📱 Responsive web interface
- 🧩 Extensible architecture for future agricultural AI modules

> **Accuracy note:** 99.1% is the validation result reported by the project. It is not a guarantee of real-world farming outcomes; model performance depends on the dataset and evaluation methodology.

## 🏗️ Architecture

```text
                    GOAgro
                       │
             ┌─────────┴─────────┐
             │                   │
        Web Interface       Flask Backend
             │                   │
             └─────────┬─────────┘
                       │
                Input Validation
                       │
                Feature Preparation
                       │
                       ▼
             ┌───────────────────┐
             │ Random Forest     │
             │ Classifier        │
             │ crop_model.pkl     │
             └─────────┬─────────┘
                       │
                       ▼
               Crop Prediction
                       │
                       ▼
             Crop / Farming Details
```

## 🤖 Machine Learning

### Model

**Random Forest Classifier**

Random Forest is an ensemble of decision trees that is well suited to structured/tabular prediction problems. In GOAgro, the model learns relationships between agricultural/environmental inputs and crop classes.

### Why Random Forest?

- Captures nonlinear relationships between features
- Works effectively with structured tabular data
- Uses an ensemble of decision trees rather than a single tree
- Provides fast inference suitable for a web application
- Can model interactions among soil and environmental variables

### Reported Result

**Validation accuracy: 99.1%**

The value above is the result reported in the project documentation.

## 🔄 Prediction Flow

1. User enters field conditions.
2. Flask receives and validates the values.
3. The values are prepared in the expected feature order.
4. The trained Random Forest model performs inference.
5. GOAgro obtains the recommended crop.
6. The application displays the crop and available supporting information.

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend** | HTML5, CSS3, JavaScript |
| **Backend** | Python, Flask |
| **Machine Learning** | Scikit-learn |
| **Data Processing** | NumPy, Pandas |
| **Model Persistence** | Joblib / `crop_model.pkl` |

## 📁 Repository Structure

```text
GOAgro/
├── app.py                  # Flask application and prediction logic
├── crop_model.pkl          # Trained crop recommendation model
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates
├── static/                 # CSS, JavaScript and assets
├── test_app.py             # Application smoke test
└── README.md               # Project documentation
```

The repository structure may evolve as new features are added.

## ⚙️ Run Locally

### Clone

```bash
git clone https://github.com/kushagra69yr/GOAgro.git
cd GOAgro
```

### Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open the local Flask address shown in the terminal.

## 🧪 Testing

Run the repository tests with:

```bash
pytest
```

The repository includes a basic application smoke test so the pytest workflow has an executable test target.

## 🗺️ Roadmap

Potential future extensions include:

- 🌿 Deep-learning plant disease detection
- ☁️ Live weather API integration
- 📍 Location-aware crop recommendations
- 🌡️ IoT soil-sensor integration
- 🌐 Multi-language support
- 📱 Mobile application
- 🛒 Farmer marketplace functionality
- 📊 Historical field analytics

These items are **future enhancements** and are not presented as currently implemented features.

## 🎯 Project Objective

GOAgro demonstrates how a machine-learning model can be integrated into a practical agriculture application to support crop-selection decisions from structured field data.

The project connects a machine-learning pipeline with a user-facing application so that model predictions can be consumed as actionable agricultural information rather than remaining only in a notebook or training environment.

## 👨‍💻 Author

### Kushagra Burman

**B.E. — Artificial Intelligence & Machine Learning**  
BMS Institute of Technology and Management, Bengaluru  
Visvesvaraya Technological University (VTU)

### Connect

- 🌐 **Portfolio:** https://kushagra69yr.github.io
- 💻 **GitHub:** https://github.com/kushagra69yr
- 💼 **LinkedIn:** https://www.linkedin.com/in/kushagra-burman-9b5740281/

## 🔗 Project Links

- **GOAgro Repository:** https://github.com/kushagra69yr/GOAgro
- **Portfolio:** https://kushagra69.github.io

## ⭐ Support

If you find GOAgro useful or interesting, consider giving the repository a ⭐ on GitHub.

---

**Built with Python, Flask and Machine Learning by Kushagra Burman.**