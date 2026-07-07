****Flight Delay Prediction System Using Machine Learning*****

##  Project Overview

Flight delays are a major challenge in the aviation industry, affecting passengers, airlines, and airport operations. This project develops a machine learning-based system that predicts whether a flight will be delayed or not using historical flight data.

The project includes data preprocessing, exploratory data analysis, feature engineering, machine learning model development, evaluation, and deployment as an interactive web application using Streamlit.

The deployed application allows users to enter flight-related details and receive a real-time delay prediction.

---

##  Live Demo

Streamlit Application:

https://flight-delay-prediction-system-using-machine-learning-a2bqwaco.streamlit.app/

---

## Objectives

- Analyze historical flight data to identify delay patterns.
- Perform data cleaning and preprocessing.
- Extract meaningful features for better prediction.
- Build and evaluate machine learning models.
- Handle class imbalance in flight delay data.
- Deploy the final prediction system using Streamlit.

---

##  Dataset

The project uses a flight delay dataset containing information related to flight operations.

### Important Features:

- **DepTime** - Actual departure time
- **Origin** - Departure airport
- **Dest** - Destination airport
- **WeatherDelay** - Delay caused due to weather

### Target Variable:

- **Delay**
  - 1 → Flight delayed (Arrival delay > 15 minutes)
  - 0 → Flight not delayed

---

#  Technologies Used

### Programming Language
- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

### Deployment

- Streamlit Community Cloud
- GitHub

---

#  Project Workflow

## 1. Data Collection

Collected historical flight data containing airline, airport, timing, and delay-related information.

---

## 2. Data Preprocessing

Performed:

- Missing value handling
- Data type conversion
- Date and time feature extraction
- Feature encoding
- Outlier removal
- Feature selection

---

#  Machine Learning Models

- Random Forest Classifier


---

#  Model Evaluation

The models were evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-score
- Confusion Matrix


---

#  Streamlit Application

The trained model was deployed as an interactive web application.

Users can provide:

- Airline details
- Departure information
- Route details
- Flight characteristics

The application predicts:

 Flight will be delayed  
or  
 Flight will not be delayed

---


