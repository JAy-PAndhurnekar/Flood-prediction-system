"""Simple training script: one Random Forest model, no comparison needed."""
import os
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

FEATURES = ["rainfall", "temperature", "humidity", "river_level", "soil_moisture", "previous_rainfall"]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "flood_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(DATASET_PATH)
    for col in FEATURES:
        df[col] = df[col].fillna(df[col].median())

    X = df[FEATURES]
    y = df["flood"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
    }
    importance = dict(sorted(
        zip(FEATURES, (model.feature_importances_ / model.feature_importances_.sum()).round(4).tolist()),
        key=lambda x: x[1], reverse=True
    ))

    joblib.dump(model, os.path.join(MODELS_DIR, "flood_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "feature_importance": importance, "features": FEATURES}, f, indent=2)

    print(f"Model trained. Accuracy: {metrics['accuracy']}%  F1: {metrics['f1']}%")
    print("Saved -> models/flood_model.pkl, models/scaler.pkl, models/metrics.json")


if __name__ == "__main__":
    train()
