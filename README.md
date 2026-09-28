# FloodAI (Easy Version) – AI/ML Flood Prediction System

A simplified, beginner-friendly version of FloodAI. Fewer moving parts, fewer setup
steps — the whole backend starts with **one command** (or one double-click on
Windows). No external APIs are used anywhere.

---

## What's Included

- **Login page** – with show/hide password toggle, "Remember me for 14 days",
  and a one-click demo admin login button
- Register (secure password hashing)
- Dashboard – latest risk, quick stats, **flood probability trend chart**
- **Type a City, Get Instant Prediction** – type/select a city name (18 built-in
  cities like Mumbai, Pune, Delhi, Chennai, etc.) and its typical local
  environmental data auto-fills, triggering an automatic prediction instantly.
  This uses only local built-in demo data — no external weather API.
- **Live Auto-Predict** – as you type environmental values (or click a Quick Demo
  button), the flood probability, risk level, model confidence, risk gauge, and
  top contributing factors update automatically — no button click needed
- **Quick Demo presets** – one-click sample data for Low / Moderate / High /
  Critical scenarios, useful for fast testing or a viva demo
- **Save to History** button — stores the prediction in the database only when
  you choose to (live preview itself doesn't spam the database)
- **AI Explanation** – shows the top 3 contributing factors behind each
  prediction, based on the trained model's feature importance
- Prediction History – table of your past saved predictions
- Alerts – auto-generated for HIGH/CRITICAL predictions
- Emergency & Awareness pages (safety info)

Removed from the full version to keep things simple: Risk Map, Forecast, full
Analytics dashboard, Admin Panel, user profile. (Ask if you want any of these
added back.)

---

## EASIEST WAY TO RUN (Windows) — One Click

1. Double-click **`start.bat`**
2. Wait for it to install everything and start the server
3. Open your browser to **http://127.0.0.1:5000**

That's it. `start.bat` automatically creates a virtual environment, installs all
libraries, and starts the app.

---

## RUNNING MANUALLY — One Command (any OS)

```bash
cd FloodAI-Easy
pip install -r requirements.txt
python app.py
```

The **first time** you run `python app.py`, it automatically:
1. Generates the demo dataset (`dataset/flood_data.csv`)
2. Trains the ML model (`models/flood_model.pkl`)
3. Sets up the SQLite database (`database/database.db`)

Every time after that, it just starts instantly (setup is skipped since the files
already exist).

Then open: **http://127.0.0.1:5000**

**Default admin login:** `admin@floodai.local` / `admin123`

---

## If `pip install` fails

Try installing one by one:
```bash
pip install Flask Werkzeug pandas numpy scikit-learn joblib
```

## If you want to start completely fresh

Delete these three files/folders' contents and run `python app.py` again:
```
dataset/flood_data.csv
models/flood_model.pkl, models/scaler.pkl, models/metrics.json
database/database.db
```

---

## Project Structure

```
FloodAI-Easy/
├── app.py                 # Everything: routes + auto-setup + server start
├── start.bat              # Windows one-click launcher
├── requirements.txt
├── utils_db.py             # Database table creation
├── ml/
│   ├── generate_dataset.py
│   └── train_model.py
├── dataset/flood_data.csv  # created automatically
├── models/                 # created automatically
├── database/                # created automatically
├── templates/               # HTML pages
└── static/css, static/js
```
