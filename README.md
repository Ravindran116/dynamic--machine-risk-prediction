# Dynamic Machine Data Management & Local Risk Prediction

## Project Description

A simple web application for managing machine information and predicting machine risk levels using a local Python Machine Learning model.

## Technologies Used

- Python
- Flask
- HTML
- CSS
- JavaScript
- SQLite
- Pandas
- Scikit-learn
- Random Forest
- Joblib

## Features

- Dynamic machine field configuration
- Add machine fields dynamically
- Support Text, Number and Dropdown fields
- Configure Required/Optional fields
- Create machine records
- View machine records
- Edit machine records
- Delete machine records
- Machine risk prediction
- Local Python Machine Learning model
- Web-based dashboard

## Machine Fields

The application initially supports:

- Machine Name
- Temperature
- Pressure
- Vibration
- Humidity

Additional fields can be added dynamically.

## Machine Learning

The application uses a locally trained Random Forest Machine Learning model.

The model predicts:

- Low Risk
- Medium Risk
- High Risk

The Machine Learning model runs locally using Python and does not use cloud-based AI services or external AI APIs and to train the dataset by using synthatic datasets.


## Application Flow

Web UI
↓
Flask Backend
↓
Machine Data
↓
Python ML Model
↓
Risk Result
↓
Web UI

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt