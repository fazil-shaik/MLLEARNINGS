import os
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, f1_score

# 1. EXTRACT Y TERMS (TARGETS)
def extract_targets(df: pd.DataFrame):
    """
    Extracts regression and classification ground truth targets.
    """
    # Continuous Target: Net delay in hours (added to scheduled arrival for final ETA)
    y_reg = df["total_arrival_delay_hours"].values
    
    # Binary Target: 1 for on-time, 0 for delayed (or vice-versa)
    y_cls = df["on_time_delivery"].values
    
    return y_reg, y_cls


# 2. FEATURE ENGINEERING & X TERMS PREPARATION
def prepare_feature_dataframe(df: pd.DataFrame, milestone: int = 0) -> pd.DataFrame:
    """
    Engineers domain features and masks downstream telemetry according to 
    the active journey milestone:
      - milestone=0: Booking (Pre-Departure)
      - milestone=1: Origin Departed
      - milestone=2: Ocean Arrived at Destination Anchorage
      - milestone=3: Customs Cleared
    """
    df = df.copy()

    # --- A. Temporal Features (from Scheduled Timestamps) ---
    sched_dep = pd.to_datetime(df["scheduled_departure_utc"])
    sched_arr = pd.to_datetime(df["scheduled_arrival_utc"])
    
    # Planned total voyage duration (hours)
    df["planned_duration_hours"] = (sched_arr - sched_dep).dt.total_seconds() / 3600.0
    df["departure_month"] = sched_dep.dt.month
    df["departure_day_of_week"] = sched_dep.dt.dayofweek
    df["departure_hour"] = sched_dep.dt.hour

    # --- B. Normalized Congestion Indices (0 to 1 scale) ---
    df["origin_congestion_norm"] = df["origin_congestion_index_0_100"] / 100.0
    df["dest_congestion_norm"] = df["destination_congestion_index_0_100"] / 100.0
    
    # --- C. Vessel & Cargo Ratios ---
    df["weight_per_teu"] = df["shipment_weight_tonnes"] / (df["vessel_capacity_teu"] + 1e-5)

    # --- D. Non-Linear Interaction Features ---
    # Weather shock scaled by destination congestion
    df["weather_x_dest_congestion"] = (df["weather_severity_0_3"] / 3.0) * df["dest_congestion_norm"]

    # --- E. Milestone-Conditioned Stage Telemetry (Prevents Data Leakage) ---
    # Stage 1: Origin Departure
    if milestone >= 1:
        df["known_departure_delay"] = df["departure_delay_hours"]
        df["known_origin_delay"] = df["origin_port_delay_hours"]
        df["known_origin_processing"] = df["origin_port_processing_hours"]
    else:
        df["known_departure_delay"] = 0.0
        df["known_origin_delay"] = 0.0
        df["known_origin_processing"] = 0.0

    # Stage 2: Ocean Voyage & Port Arrival
    if milestone >= 2:
        df["known_sea_transit_delay"] = df["sea_transit_delay_hours"]
        df["cum_delay_at_port"] = df["known_departure_delay"] + df["known_sea_transit_delay"]
        # Non-linear berth window forfeiture risk
        df["berth_risk_factor"] = (df["cum_delay_at_port"] > 10.0).astype(float) * df["dest_congestion_norm"]
    else:
        df["known_sea_transit_delay"] = 0.0
        df["cum_delay_at_port"] = df["known_departure_delay"]
        df["berth_risk_factor"] = 0.0

    # Stage 3: Port Berthing & Customs
    if milestone >= 3:
        df["known_dest_port_delay"] = df["destination_port_delay_hours"]
        df["known_customs_delay"] = df["customs_delay_hours"]
        df["known_dest_processing"] = df["destination_port_processing_hours"]
    else:
        df["known_dest_port_delay"] = 0.0
        df["known_customs_delay"] = 0.0
        df["known_dest_processing"] = 0.0

    return df


# 3. DEFINE FEATURE COLUMN GROUPS & PREPROCESSOR
CATEGORICAL_FEATURES = [
    "origin_port",
    "origin_country",
    "destination_port",
    "destination_country",
    "cargo_type",
    "weather_condition",
]

NUMERICAL_FEATURES = [
    "planned_duration_hours",
    "departure_month",
    "departure_day_of_week",
    "departure_hour",
    "origin_congestion_norm",
    "dest_congestion_norm",
    "weather_severity_0_3",
    "vessel_capacity_teu",
    "shipment_weight_tonnes",
    "weight_per_teu",
    "weather_x_dest_congestion",
    "known_departure_delay",
    "known_origin_delay",
    "known_origin_processing",
    "known_sea_transit_delay",
    "cum_delay_at_port",
    "berth_risk_factor",
    "known_dest_port_delay",
    "known_customs_delay",
    "known_dest_processing",
]

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


# 4. PREDICTION, EVALUATION & EXPLAINABILITY FUNCTIONS
def predict_shipment_eta_and_risk(model_reg, model_cls, preprocessor, new_shipment_df, milestone=0):
    processed_df = prepare_feature_dataframe(new_shipment_df, milestone=milestone)
    X_new = preprocessor.transform(processed_df)
    
    pred_delay_hours = float(model_reg.predict(X_new)[0])
    pred_delay_prob = float(model_cls.predict_proba(X_new)[0, 1]) * 100.0
    
    sched_arrival = pd.to_datetime(new_shipment_df["scheduled_arrival_utc"].iloc[0])
    pred_final_eta = sched_arrival + timedelta(hours=pred_delay_hours)
    
    if pred_delay_hours <= 1.0 and pred_delay_prob < 35.0:
        severity = "On-Time / Early"
    elif pred_delay_hours <= 8.0:
        severity = "Minor Delay (Low Risk)"
    elif pred_delay_hours <= 24.0:
        severity = "Moderate Delay (Medium Risk)"
    else:
        severity = "Critical Delay (High Risk)"
        
    return {
        "scheduled_arrival": sched_arrival.strftime("%Y-%m-%d %H:%M UTC"),
        "predicted_final_eta": pred_final_eta.strftime("%Y-%m-%d %H:%M UTC"),
        "predicted_delay_hours": round(pred_delay_hours, 1),
        "delay_probability_pct": round(pred_delay_prob, 1),
        "severity_tier": severity
    }


def evaluate_system(model_reg, model_cls, X_test, y_test_reg, y_test_cls):
    pred_reg = model_reg.predict(X_test)
    pred_cls_prob = model_cls.predict_proba(X_test)[:, 1]
    pred_cls = (pred_cls_prob >= 0.5).astype(int)
    
    mae = mean_absolute_error(y_test_reg, pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, pred_reg))
    r2 = r2_score(y_test_reg, pred_reg)
    f1 = f1_score(y_test_cls, pred_cls, zero_division=0)
    auc = roc_auc_score(y_test_cls, pred_cls_prob)
    baseline_mae = mean_absolute_error(y_test_reg, np.zeros_like(y_test_reg))
    
    print("\n" + "=" * 55)
    print("AI MODEL EVALUATION & BENCHMARK REPORT")
    print("=" * 55)
    print(f"Mean Absolute Error (MAE):      {mae:.2f} hours (Baseline: {baseline_mae:.2f} hours)")
    print(f"Root Mean Squared Error (RMSE):  {rmse:.2f} hours")
    print(f"R-Squared (R2) Variance Score:  {r2:.3f}")
    print(f"Classification F1-Score:        {f1:.3f}")
    print(f"ROC-AUC Score:                  {auc:.3f}")
    print(f"Error Reduction vs Timetable:   {((baseline_mae - mae) / baseline_mae) * 100:.1f}%")
    print("=" * 55)


def explain_delay_factors(model_reg, preprocessor, shipment_row):
    num_cols = NUMERICAL_FEATURES
    cat_cols = list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = num_cols + cat_cols
    
    importances = model_reg.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values(by="importance", ascending=False).reset_index(drop=True)
    
    print("\nTop 5 Global Delay Drivers:")
    print(fi_df.head(5).to_string(index=False))
    
    c_origin = float(shipment_row.get("departure_delay_hours", 0.0))
    c_transit = float(shipment_row.get("sea_transit_delay_hours", 0.0))
    c_port = float(shipment_row.get("destination_port_delay_hours", 0.0))
    c_customs = float(shipment_row.get("customs_delay_hours", 0.0))
    
    stage_breakdown = {
        "Origin Departure Lag": c_origin,
        "Ocean Weather / Sea Transit": c_transit,
        "Destination Port Congestion / Berth Queue": c_port,
        "Customs Inspection Dwell": c_customs
    }
    primary_cause = max(stage_breakdown, key=stage_breakdown.get)
    print(f"\nShipment Primary Root Cause: {primary_cause} (+{stage_breakdown[primary_cause]:.1f} hours)")


# 5. COMPLETE EXECUTION FLOW
if __name__ == "__main__":
    # Check if dataset exists, fallback to data/shipments_dataset.csv if ./data.csv doesn't exist
    dataset_path = "./data.csv" if os.path.exists("./data.csv") else "data/shipments_dataset.csv"
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # 1. Extract Targets (Y)
    y_reg, y_cls = extract_targets(df)

    # 2. Extract Features (X)
    df_engineered = prepare_feature_dataframe(df, milestone=0)
    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(df_engineered)

    # 3. Train/Test Split
    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42
    )

    # 4. Train Both Models (Regressor for ETA + Classifier for Delay Risk)
    print("Training Gradient Boosting Regressor (ETA Magnitude)...")
    model_reg = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model_reg.fit(X_train, y_train_reg)

    print("Training Random Forest Classifier (Delay Likelihood)...")
    model_cls = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_cls.fit(X_train, y_train_cls)

    # 5. Evaluate Performance
    evaluate_system(model_reg, model_cls, X_test, y_test_reg, y_test_cls)

    # 6. Predict for a Sample In-Flight Shipment
    sample_shipment = df.iloc[[0]]
    prediction = predict_shipment_eta_and_risk(model_reg, model_cls, preprocessor, sample_shipment, milestone=1)
    
    print("\nSample In-Flight Shipment Prediction:")
    for key, value in prediction.items():
        print(f"  • {key}: {value}")

    # 7. Explain Delay Drivers
    explain_delay_factors(model_reg, preprocessor, sample_shipment.iloc[0])