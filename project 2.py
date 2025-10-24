# ================================================================
# EMOTIONAL HEALTH EARLY WARNING SYSTEM (Prototype)
# ================================================================
# Author: Abdurrahmaan Tayob
# Description: Data Science project prototype that simulates
# passive digital-biomarker data, trains a predictive model,
# and demonstrates alert logic for early detection of emotional
# distress using simulated time-series data.
# ================================================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import joblib
import os

# ------------------------------------------------
# 1. SIMULATE SYNTHETIC DATA
# ------------------------------------------------
np.random.seed(42)

def simulate_user_days(n_users=50, days=90):
    rows = []
    for user in range(n_users):
        baseline_sleep = np.random.normal(7, 0.6)
        baseline_steps = np.random.normal(7000, 1500)
        baseline_phone = np.random.normal(180, 40)
        baseline_typing = np.random.normal(250, 30)
        baseline_hrv = np.random.normal(50, 10)

        for d in range(days):
            date = datetime(2025, 1, 1) + timedelta(days=d)

            # daily variation
            sleep = max(3, np.random.normal(baseline_sleep, 1.0))
            steps = max(0, np.random.normal(baseline_steps, 2000))
            phone = max(10, np.random.normal(baseline_phone, 60))
            typing = max(50, np.random.normal(baseline_typing, 40))
            hrv = max(10, np.random.normal(baseline_hrv, 12))
            voice_sentiment = np.clip(np.random.normal(0.1, 0.6) - (phone-180)/6000, -1, 1)
            weather_stress = np.random.binomial(1, 0.05) * np.random.uniform(0.3, 1.0)

            # latent stress
            stress_latent = (
                0.5 * (7 - sleep) / 4.0 +
                0.3 * (phone - 150) / 400.0 +
                0.2 * (7000 - steps) / 8000.0 +
                0.25 * (40 - hrv) / 80.0 +
                0.4 * (-voice_sentiment) +
                0.4 * weather_stress +
                np.random.normal(0, 0.15)
            )

            # distress probability
            prob_distress = 1 / (1 + np.exp(-3.2 * (stress_latent - 0.2)))
            distress = np.random.binomial(1, np.clip(prob_distress, 0, 0.95))
            mood_cont = np.clip(5 - (stress_latent * 2.5) + np.random.normal(0, 0.5), 1, 5)
            mood = int(round(mood_cont))

            rows.append({
                "user_id": f"user_{user:03d}",
                "date": date,
                "sleep_hours": round(sleep,2),
                "steps": int(steps),
                "phone_minutes": int(phone),
                "typing_speed": int(typing),
                "hrv": round(hrv,1),
                "voice_sentiment": round(voice_sentiment,3),
                "weather_stress": round(weather_stress,3),
                "distress": int(distress),
                "self_report_mood": mood
            })
    return pd.DataFrame(rows)

df = simulate_user_days(n_users=40, days=75)
print("✅ Data simulated:", df.shape)
print(df.head())

# ------------------------------------------------
# 2. FEATURE ENGINEERING
# ------------------------------------------------
df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])

# rolling features
for window in [3, 7]:
    df[f"sleep_roll_{window}"] = df.groupby("user_id")["sleep_hours"].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
    df[f"steps_roll_{window}"] = df.groupby("user_id")["steps"].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
    df[f"phone_roll_{window}"] = df.groupby("user_id")["phone_minutes"].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
    df[f"hrv_roll_{window}"] = df.groupby("user_id")["hrv"].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
    df[f"typing_roll_{window}"] = df.groupby("user_id")["typing_speed"].rolling(window, min_periods=1).mean().reset_index(0, drop=True)

# target: next-day distress
df["distress_next"] = df.groupby("user_id")["distress"].shift(-1)
df = df.dropna(subset=["distress_next"])
df["distress_next"] = df["distress_next"].astype(int)

feature_cols = [
    "sleep_hours","steps","phone_minutes","typing_speed","hrv",
    "voice_sentiment","weather_stress",
    "sleep_roll_3","sleep_roll_7","steps_roll_3","steps_roll_7",
    "phone_roll_3","phone_roll_7","hrv_roll_3","hrv_roll_7"
]

# split by user
users = df["user_id"].unique()
train_users, test_users = train_test_split(users, test_size=0.2, random_state=42)
X_train = df[df["user_id"].isin(train_users)][feature_cols]
y_train = df[df["user_id"].isin(train_users)]["distress_next"]
X_test  = df[df["user_id"].isin(test_users)][feature_cols]
y_test  = df[df["user_id"].isin(test_users)]["distress_next"]

# ------------------------------------------------
# 3. TRAIN MODEL
# ------------------------------------------------
model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "ROC AUC": roc_auc_score(y_test, y_proba)
}

print("\n📊 Model Performance:")
for k,v in metrics.items():
    print(f"{k}: {v:.3f}")

# confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ------------------------------------------------
# 4. SAVE MODEL & DATA
# ------------------------------------------------
os.makedirs("output", exist_ok=True)
joblib.dump({"model": model, "feature_cols": feature_cols}, "output/emotional_ews_model.pkl")
df.to_csv("output/emotional_ews_sample.csv", index=False)

print("\n💾 Files saved in /output:")
print("- emotional_ews_model.pkl")
print("- emotional_ews_sample.csv")

# ------------------------------------------------
# 5. VISUALIZATION
# ------------------------------------------------
# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {metrics['ROC AUC']:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Emotional Health Early Warning System")
plt.legend()
plt.show()

# Feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(7,4))
plt.bar(importances.index[:10], importances.values[:10])
plt.xticks(rotation=45, ha='right')
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 6. ALERT LOGIC DEMONSTRATION
# ------------------------------------------------
def generate_alert(feature_vector, threshold=0.45):
    """Predict distress probability and generate alert message."""
    proba = model.predict_proba(feature_vector.values.reshape(1,-1))[:,1][0]
    diffs = (feature_vector - X_train.median()).abs().sort_values(ascending=False)
    top_features = diffs.index[:3].tolist()
    return {
        "predicted_distress_proba": round(float(proba), 3),
        "alert": bool(proba >= threshold),
        "top_feature_deviations": top_features
    }

example_user = test_users[0]
user_df = df[df["user_id"] == example_user].copy().sort_values("date")
print(f"\n🔎 Example alerts for {example_user}:")
for _, row in user_df.tail(5).iterrows():
    alert = generate_alert(row[feature_cols])
    print(row["date"].date(), "→", alert)

# ------------------------------------------------
# 7. HOW TO LOAD & USE THE MODEL
# ------------------------------------------------
# Example of reloading the model for future use:
"""
import joblib, pandas as pd

data = joblib.load("output/emotional_ews_model.pkl")
model = data["model"]
feature_cols = data["feature_cols"]

# Load or create a new day's data
new_day = pd.DataFrame([{
    "sleep_hours": 6.2, "steps": 5200, "phone_minutes": 220, "typing_speed": 240,
    "hrv": 40, "voice_sentiment": -0.3, "weather_stress": 0.8,
    "sleep_roll_3": 6.8, "sleep_roll_7": 7.0, "steps_roll_3": 6000,
    "steps_roll_7": 6500, "phone_roll_3": 210, "phone_roll_7": 200,
    "hrv_roll_3": 42, "hrv_roll_7": 44
}])

# Predict
proba = model.predict_proba(new_day[feature_cols])[:,1][0]
print("Predicted distress probability:", round(proba,3))
"""
# ================================================================
