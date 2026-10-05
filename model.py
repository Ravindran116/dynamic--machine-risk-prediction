import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# --------------------------------
# 1. Load training data
# --------------------------------

data = pd.read_csv("training_data.csv")


# --------------------------------
# 2. Separate input and output
# --------------------------------

X = data[
    [
        "Temperature",
        "Pressure",
        "Vibration",
        "Humidity"
    ]
]

y = data["Risk"]


# --------------------------------
# 3. Convert Vibration text
#    Low / Medium / High
#    into numbers automatically
# --------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "vibration",
            OneHotEncoder(handle_unknown="ignore"),
            ["Vibration"]
        )
    ],
    remainder="passthrough"
)


# --------------------------------
# 4. Create ML model
# --------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)


# --------------------------------
# 5. Split training data
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------
# 6. Train the ML model
# --------------------------------

model.fit(X_train, y_train)


# --------------------------------
# 7. Test model accuracy
# --------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Model trained successfully!")
print("Model Accuracy:", accuracy)


# --------------------------------
# 8. Save trained model
# --------------------------------

joblib.dump(
    model,
    "machine_risk_model.pkl"
)

print("Model saved as machine_risk_model.pkl")