import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss

def load_dataset(session_type="RAC"):
    df = pd.read_csv("data/processed/dataset_ml.csv", low_memory=False)
    return df[df["session_type"] == session_type].reset_index(drop=True)

def split_data(df):
    train = df[df["season_year"] <= 2024].reset_index(drop=True)
    val = df[df["season_year"] == 2025].reset_index(drop=True)
    test = df[df["season_year"] == 2026].reset_index(drop=True)
    return train, val, test

def get_features_target(df, session_type):
    base_cols = ["grid_position", "is_front_row", "is_wet", "rider_last3_avg", "rider_last5_avg", "dnf_ratio_last5", "dnf_ratio_last10", "rider_avg_gain", "rider_circuit_avg", "constructor_circuit_avg", "constructor_last3_avg", "constructor_last5_avg", "circuit_dnf_rate", "clean_finish_streak", "fp_avg_pos", "pr_pos", "fp_vs_grid_delta"]
    if session_type == "RAC":
        cols = base_cols + ["rider_last3_avg_rac", "sprint_pos_same_weekend"]
    elif session_type == "SPR":
        cols = base_cols + ["rider_last3_avg_spr"]
    else:
        raise ValueError("invalid session type")
    return df[cols], df["position_cleaned"], df["is_dnf"]

def get_dnf_features(df):
    dnf_cols = ["grid_position", "is_front_row", "is_wet", "dnf_ratio_last5", "dnf_ratio_last10", "rider_avg_gain", "constructor_circuit_avg", "constructor_last5_avg", "circuit_dnf_rate", "clean_finish_streak", "fp_vs_grid_delta"]
    return df[dnf_cols], df["is_dnf"]

def train_models(df_train, session_type):
    train_fin = df_train[df_train["is_dnf"] == 0].reset_index(drop=True)
    X_fin, y_fin, _ = get_features_target(train_fin, session_type)
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=3, random_state=42)
    reg.fit(X_fin, y_fin)
    X_dnf, y_dnf = get_dnf_features(df_train)
    clf = HistGradientBoostingClassifier(max_iter=75, learning_rate=0.015, max_depth=2, min_samples_leaf=60, l2_regularization=5.0, random_state=42)
    clf.fit(X_dnf, y_dnf)
    return reg, clf

def evaluate(reg, clf, df_eval, session_type, label="Validation"):
    eval_fin = df_eval[df_eval["is_dnf"] == 0].reset_index(drop=True)
    X_fin, y_fin, _ = get_features_target(eval_fin, session_type)
    preds_fin = reg.predict(X_fin)
    mae = mean_absolute_error(y_fin, preds_fin)
    std_dev = float(np.std(y_fin - preds_fin))
    X_reg_all, _, _ = get_features_target(df_eval, session_type)
    X_dnf, y_dnf = get_dnf_features(df_eval)
    dnf_probs = clf.predict_proba(X_dnf)[:, 1]
    auc = roc_auc_score(y_dnf, dnf_probs)
    brier = brier_score_loss(y_dnf, dnf_probs)
    eval_df = df_eval[["session_id", "rider_name", "position_cleaned"]].copy()
    eval_df["pred_score"] = reg.predict(X_reg_all) + dnf_probs * 15.0
    eval_df["pred_rank"] = eval_df.groupby("session_id")["pred_score"].rank(method="first")
    podium_hits = [len(set(g[g["position_cleaned"] <= 3]["rider_name"]) & set(g[g["pred_rank"] <= 3]["rider_name"])) for _, g in eval_df.groupby("session_id")]
    print(f"[{label}] MAE (arrivati): {mae:.2f} | Std Dev: {std_dev:.2f} | DNF AUC: {auc:.3f} | DNF Brier: {brier:.3f} | Podi: {np.mean(podium_hits):.2f}/3")
    return std_dev

def save_bundle(reg, clf, std_dev, session_type):
    os.makedirs("models", exist_ok=True)
    path = f"models/model_{session_type.lower()}.pkl"
    joblib.dump({"reg": reg, "clf": clf, "std_dev": std_dev}, path)
    print(f"Bundle salvato in {path}")

if __name__ == "__main__":
    for session_type in ["RAC", "SPR"]:
        print(f"\ntraining {session_type} model...")
        df = load_dataset(session_type)
        train, val, test = split_data(df)
        reg, clf = train_models(train, session_type)
        std_dev = evaluate(reg, clf, val, session_type, "Validation 2025")
        evaluate(reg, clf, test, session_type, "Test 2026")
        save_bundle(reg, clf, std_dev, session_type)