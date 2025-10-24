Of course — here’s a clean, professional **GitHub project summary** with **no emojis**, ready for your `README.md` file:

---

# Emotional Health Early Warning System (Prototype)

## Project Overview

This project is a data science prototype that simulates and predicts early signs of emotional distress using passive digital biomarkers such as sleep, activity, phone usage, and heart rate variability (HRV).
It demonstrates how data science and machine learning can be applied to build early mental well-being monitoring systems without using sensitive personal data.

The system uses synthetic data to mimic patterns from wearable devices and smartphone sensors, then trains a model to predict next-day emotional distress risk for each user.

---

## Key Features

* **Self-contained:** No external data required; the script generates its own synthetic dataset.
* **Data Simulation:** Creates realistic time-series data for multiple users (sleep, steps, HRV, phone use, etc.).
* **Feature Engineering:** Adds rolling averages (3-day and 7-day windows) to capture behavioral trends.
* **Predictive Modeling:** Trains a Random Forest classifier to forecast next-day emotional distress.
* **Model Evaluation:** Reports key metrics including Accuracy, F1 Score, Precision, Recall, and ROC AUC.
* **Explainable Alerts:** Generates alert messages highlighting which behaviors deviated most from normal.
* **Visualization:** Displays ROC curve and feature importance plots.
* **Model Persistence:** Saves both the model and dataset for reuse (`.pkl` and `.csv` files).

---

## Tech Stack

* Python 3.10+
* Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`

---

## Model Summary

* **Algorithm:** Random Forest Classifier
* **Target Variable:** `distress_next` — predicts whether a user experiences emotional distress the following day
* **Input Features:** Sleep hours, steps, phone usage, HRV, typing speed, voice sentiment, and rolling averages
* **Performance (on synthetic data):**

  * Accuracy: ~0.70
  * ROC AUC: ~0.57
  * Precision: ~0.40
  * Recall: ~0.06

*(These results reflect a realistic early-stage prototype using synthetic data — not clinical performance.)*

---

## Outputs

When you run the script, it automatically creates an `output/` folder containing:

* `emotional_ews_sample.csv` — simulated dataset (all users × all days)
* `emotional_ews_model.pkl` — trained model file (ready for reuse)

Example predictions and alert logic are printed in the terminal:

```
2025-03-15 → {'predicted_distress_proba': 0.52, 'alert': True, 'top_feature_deviations': ['sleep_hours', 'hrv', 'phone_minutes']}
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/emotional-ews.git
cd emotional-ews

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib joblib

# 3. Run the script
python emotional_ews.py
```

---



