"""
FloodAI (Easy Version) - single command startup.
Run:  python app.py
On first run this automatically creates the demo dataset, trains the
ML model, and sets up the SQLite database - no separate setup commands
needed.
"""
import os
import json
import sqlite3
import subprocess
import sys
from datetime import timedelta
from functools import wraps

import joblib
from flask import Flask, render_template, request, redirect, url_for, session, flash, g, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "flood_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "flood_data.csv")

FEATURES = ["rainfall", "temperature", "humidity", "river_level", "soil_moisture", "previous_rainfall"]

app = Flask(__name__)
app.secret_key = "floodai-easy-dev-secret"
app.permanent_session_lifetime = timedelta(days=14)

RECOMMENDATIONS = {
    "LOW": "Flood risk is currently low. Continue normal monitoring.",
    "MODERATE": "Moderate flood risk detected. Monitor environmental conditions.",
    "HIGH": "High flood risk detected. Avoid low-lying areas and monitor official warnings.",
    "CRITICAL": "Critical flood risk detected. Move to a safer location and follow emergency instructions."
}

METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

_model, _scaler, _metrics = None, None, None

# Quick-fill demo presets so the app can be tested with one click, no typing needed
QUICK_PRESETS = {
    "low": {"location": "Nashik", "rainfall": 8, "previous_rainfall": 6, "temperature": 25,
            "humidity": 45, "river_level": 1.0, "soil_moisture": 22},
    "moderate": {"location": "Pune", "rainfall": 32, "previous_rainfall": 20, "temperature": 27,
                 "humidity": 62, "river_level": 2.2, "soil_moisture": 42},
    "high": {"location": "Mumbai", "rainfall": 62, "previous_rainfall": 55, "temperature": 28,
             "humidity": 79, "river_level": 3.8, "soil_moisture": 68},
    "critical": {"location": "Nagpur", "rainfall": 95, "previous_rainfall": 110, "temperature": 29,
                 "humidity": 89, "river_level": 5.6, "soil_moisture": 88},
}

# Local built-in city environmental profiles (demo data, no external weather API).
# Typing/selecting a city auto-fills these values and triggers an automatic prediction.
CITY_PROFILES = {
    "Mumbai":      {"rainfall": 62, "previous_rainfall": 55, "temperature": 28, "humidity": 79, "river_level": 3.8, "soil_moisture": 68},
    "Pune":        {"rainfall": 30, "previous_rainfall": 20, "temperature": 27, "humidity": 60, "river_level": 2.0, "soil_moisture": 40},
    "Nagpur":      {"rainfall": 95, "previous_rainfall": 110, "temperature": 29, "humidity": 89, "river_level": 5.6, "soil_moisture": 88},
    "Nashik":      {"rainfall": 8,  "previous_rainfall": 6,  "temperature": 25, "humidity": 45, "river_level": 1.0, "soil_moisture": 22},
    "Delhi":       {"rainfall": 20, "previous_rainfall": 15, "temperature": 32, "humidity": 55, "river_level": 1.8, "soil_moisture": 30},
    "Mumbai Suburban": {"rainfall": 70, "previous_rainfall": 65, "temperature": 27, "humidity": 82, "river_level": 4.1, "soil_moisture": 72},
    "Chennai":     {"rainfall": 55, "previous_rainfall": 48, "temperature": 31, "humidity": 76, "river_level": 3.2, "soil_moisture": 60},
    "Kolkata":     {"rainfall": 80, "previous_rainfall": 70, "temperature": 30, "humidity": 84, "river_level": 4.6, "soil_moisture": 75},
    "Bengaluru":   {"rainfall": 18, "previous_rainfall": 12, "temperature": 24, "humidity": 58, "river_level": 1.3, "soil_moisture": 33},
    "Hyderabad":   {"rainfall": 25, "previous_rainfall": 18, "temperature": 29, "humidity": 56, "river_level": 1.6, "soil_moisture": 35},
    "Ahmedabad":   {"rainfall": 15, "previous_rainfall": 10, "temperature": 33, "humidity": 48, "river_level": 1.2, "soil_moisture": 25},
    "Surat":       {"rainfall": 88, "previous_rainfall": 95, "temperature": 28, "humidity": 85, "river_level": 5.2, "soil_moisture": 82},
    "Patna":       {"rainfall": 72, "previous_rainfall": 68, "temperature": 29, "humidity": 80, "river_level": 4.3, "soil_moisture": 70},
    "Guwahati":    {"rainfall": 98, "previous_rainfall": 120, "temperature": 27, "humidity": 90, "river_level": 6.1, "soil_moisture": 91},
    "Kochi":       {"rainfall": 65, "previous_rainfall": 58, "temperature": 27, "humidity": 83, "river_level": 3.6, "soil_moisture": 65},
    "Jaipur":      {"rainfall": 10, "previous_rainfall": 7,  "temperature": 34, "humidity": 40, "river_level": 0.8, "soil_moisture": 18},
    "Lucknow":     {"rainfall": 40, "previous_rainfall": 35, "temperature": 30, "humidity": 65, "river_level": 2.4, "soil_moisture": 45},
    "Bhopal":      {"rainfall": 35, "previous_rainfall": 28, "temperature": 28, "humidity": 62, "river_level": 2.1, "soil_moisture": 42},
}

BASELINES = {"rainfall": 15, "temperature": 27, "humidity": 55, "river_level": 1.5,
             "soil_moisture": 30, "previous_rainfall": 15}


def classify_risk(p):
    if p <= 30:
        return "LOW"
    elif p <= 60:
        return "MODERATE"
    elif p <= 80:
        return "HIGH"
    return "CRITICAL"


def get_model():
    global _model, _scaler
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    return _model, _scaler


def get_metrics():
    global _metrics
    if _metrics is None and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)
    return _metrics or {}


def predict(input_dict):
    import pandas as pd
    model, scaler = get_model()
    X = pd.DataFrame([[input_dict[f] for f in FEATURES]], columns=FEATURES)
    proba = model.predict_proba(scaler.transform(X))[0]
    probability = round(float(proba[1]) * 100, 2)
    confidence = round(float(max(proba)) * 100, 2)
    return probability, classify_risk(probability), confidence


def explain(input_dict):
    """Simple explanation using the model's feature importance (viva-friendly)."""
    importance = get_metrics().get("feature_importance", {f: 1 / len(FEATURES) for f in FEATURES})
    scored = []
    for f in FEATURES:
        deviation = max(input_dict[f] - BASELINES[f], 0)
        scored.append((f, importance.get(f, 0) * deviation))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [f.replace("_", " ").title() for f, s in scored[:3] if s > 0]
    return top or ["Overall environmental conditions"]


# ---------------------------------------------------------------- DATABASE
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db:
        db.close()


def login_required(view):
    @wraps(view)
    def wrapped(*a, **kw):
        if not session.get("user_id"):
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view(*a, **kw)
    return wrapped


# ---------------------------------------------------------------- AUTH
@app.route("/")
def index():
    return redirect(url_for("dashboard") if session.get("user_id") else url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if not name or not email or not password:
            flash("All fields are required.", "danger")
        elif password != confirm:
            flash("Passwords do not match.", "danger")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
        else:
            db = get_db()
            if db.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone():
                flash("Email already registered.", "danger")
            else:
                db.execute("INSERT INTO users (name, email, password, role) VALUES (?,?,?, 'user')",
                           (name, email, generate_password_hash(password)))
                db.commit()
                flash("Account created. Please log in.", "success")
                return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember = request.form.get("remember")
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["name"] = user["name"]
            session.permanent = bool(remember)
            flash(f"Welcome, {user['name']}!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------- PAGES
@app.route("/dashboard")
@login_required
def dashboard():
    db = get_db()
    recent = db.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 1",
                         (session["user_id"],)).fetchone()
    stats = db.execute(
        """SELECT COUNT(*) total,
                  SUM(CASE WHEN risk_level IN ('HIGH','CRITICAL') THEN 1 ELSE 0 END) risky,
                  AVG(flood_probability) avg_prob
           FROM predictions WHERE user_id=?""", (session["user_id"],)).fetchone()
    trend_rows = db.execute(
        """SELECT flood_probability, rainfall, created_at FROM predictions
           WHERE user_id=? ORDER BY created_at DESC LIMIT 10""", (session["user_id"],)
    ).fetchall()
    trend_rows = list(reversed(trend_rows))
    return render_template("dashboard.html", recent=recent, stats=stats, trend_rows=trend_rows)


@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():
    result = None
    if request.method == "POST":
        try:
            location = request.form.get("location", "Unknown").strip() or "Unknown"
            data = {f: float(request.form.get(f, 0)) for f in FEATURES}
            for f in FEATURES:
                if data[f] < 0:
                    raise ValueError("Values cannot be negative.")
            probability, risk, confidence = predict(data)
            db = get_db()
            db.execute(
                """INSERT INTO predictions (user_id, location, rainfall, temperature, humidity,
                   river_level, soil_moisture, previous_rainfall, flood_probability, risk_level,
                   model_confidence)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (session["user_id"], location, data["rainfall"], data["temperature"], data["humidity"],
                 data["river_level"], data["soil_moisture"], data["previous_rainfall"], probability, risk,
                 confidence)
            )
            if risk in ("HIGH", "CRITICAL"):
                db.execute("INSERT INTO alerts (location, risk_level, probability, message) VALUES (?,?,?,?)",
                           (location, risk, probability, RECOMMENDATIONS[risk]))
            db.commit()
            result = {"location": location, "probability": probability, "risk": risk,
                      "confidence": confidence, "recommendation": RECOMMENDATIONS[risk],
                      "factors": explain(data), "saved": True}
            flash("Prediction saved to history.", "success")
        except ValueError as e:
            flash(str(e), "danger")
    return render_template("prediction.html", result=result, presets=QUICK_PRESETS, cities=CITY_PROFILES)


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    """Used for automatic/live prediction as the user types — does not save to history."""
    try:
        payload = request.get_json(force=True)
        data = {f: float(payload.get(f, 0)) for f in FEATURES}
        for f in FEATURES:
            if data[f] < 0:
                return jsonify({"error": "Values cannot be negative."}), 400
        probability, risk, confidence = predict(data)
        return jsonify({
            "probability": probability,
            "risk": risk,
            "confidence": confidence,
            "recommendation": RECOMMENDATIONS[risk],
            "factors": explain(data),
        })
    except (TypeError, ValueError):
        return jsonify({"error": "Please enter valid numbers in every field."}), 400
    except (FileNotFoundError, OSError):
        return jsonify({"error": "Model not ready yet. Please wait a moment and try again."}), 503


@app.route("/history")
@login_required
def history():
    db = get_db()
    rows = db.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 50",
                       (session["user_id"],)).fetchall()
    return render_template("history.html", rows=rows)


@app.route("/alerts")
@login_required
def alerts():
    db = get_db()
    rows = db.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT 30").fetchall()
    return render_template("alerts.html", alerts=rows)


@app.route("/emergency")
@login_required
def emergency():
    return render_template("emergency.html")


@app.route("/awareness")
@login_required
def awareness():
    return render_template("awareness.html")


# ---------------------------------------------------------------- AUTO SETUP
def auto_setup():
    """Runs automatically so the whole app starts with a single command."""
    os.makedirs(os.path.join(BASE_DIR, "dataset"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "database"), exist_ok=True)

    if not os.path.exists(DATASET_PATH):
        print("[setup] Generating demo dataset...")
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "ml", "generate_dataset.py")],
                        cwd=BASE_DIR, check=True)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("[setup] Training ML model (first run only, takes a few seconds)...")
        subprocess.run([sys.executable, os.path.join(BASE_DIR, "ml", "train_model.py")],
                        cwd=BASE_DIR, check=True)

    if not os.path.exists(DB_PATH):
        print("[setup] Initializing database...")
        from utils_db import init_db
        init_db()

    print("[setup] Ready. Default admin login -> admin@floodai.local / admin123")


if __name__ == "__main__":
    auto_setup()
    print("\nStarting FloodAI at http://127.0.0.1:5000\n")
    app.run(debug=False, host="127.0.0.1", port=5000)
