# Diabetes Detection Project

A full-stack-style machine learning project for diabetes risk prediction using patient health indicators. This repository contains a Streamlit web app that loads a trained model and predicts whether a person is likely diabetic based on key health features like BMI, blood glucose, insulin, and HbA1c values.

## Project Overview

The goal of this project is to build an accessible and user-friendly prediction system that helps estimate diabetes risk from health metrics. It includes:

- A trained machine learning pipeline for classification
- A Streamlit dashboard for interactive prediction
- A real-time risk probability display
- Visual analytics for better understanding of the prediction result
- Clean project organization for future extension and deployment

## Features

- Predict diabetes risk from user inputs
- Interactive sidebar form for patient data
- Real-time probability score
- Classification result: Diabetic / Non-Diabetic
- Risk-based alerting and recommendations
- Visual charts using Plotly
- Project structure for scalability and maintainability

## Repository Structure

```text
Diabeties_Detection_Project/
├── App.py                             # Main Streamlit application
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── pipeline.pkl                       # Trained prediction model
├── diabetes-prediction-dataset.csv    # Training dataset
├── diabetes-prediction-dataset_.csv   # Alternate dataset copy
├── Diabeties_detection.ipynb          # Notebook for model training and analysis
├── ui_utils.py                        # UI helper functions and styling
├── .devcontainer/                    # Dev container setup
├── .ipynb_checkpoints/               # Jupyter checkpoint files
├── __pycache__/                      # Python cache files
├── catboost_info/                    # CatBoost metadata
└── .github/                          # GitHub automation/config
```

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn
- Plotly
- CatBoost
- Jupyter Notebook

## Model Details

The application uses a trained machine learning pipeline that includes preprocessing and classification logic. The model predicts the probability of diabetes based on features including:

- Age
- HighBP
- HighChol
- Smoker
- Sex
- BMI
- blood_glucose_level
- Insulin
- HbA1c_level

## Installation

1. Clone the repository:

```bash
git clone https://github.com/junaidulhassan/Diabeties_Detection_Project.git
cd Diabeties_Detection_Project
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

Start the Streamlit application:

```bash
streamlit run App.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## Usage

1. Open the app in the browser.
2. Enter patient health information in the sidebar.
3. View the predicted diabetes probability and classification.
4. Review the risk alert and visualization to understand the result.

## Example Input Fields

- Age range
- High blood pressure status
- High cholesterol status
- Smoking status
- Gender
- BMI
- Blood glucose level
- Insulin level
- HbA1c level

## Project Goals

This project demonstrates:

- End-to-end ML workflow for healthcare prediction
- Use of binary classification for disease risk estimation
- Deployment of a model via a web app
- Visual explanation of model output for end users

## Data Source

The project uses a diabetes prediction dataset containing patient health parameters and a binary diabetes label. It is intended for educational and research-oriented prediction experiments.

## Limitations

- This is not a medical diagnosis tool.
- Predictions should not replace professional medical evaluation.
- Model accuracy and reliability depend on the underlying training data.

## Disclaimer

This project is for educational and research purposes only. It should not be used for clinical decision-making or as a substitute for medical advice from a qualified healthcare professional.

## Author

Junaid Ul Hassan

GitHub: https://github.com/junaidulhassan

## License

This repository is currently provided for educational use. Please check if a formal license file is added later before using it in production or public distribution.
