import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss

def load_dataset(session_type="RAC"):
    df = pd.read_csv("data/processed/dataset_ml.csv", low_memory=False)
    return df[df["session_type"] == session_type].reset_index(drop=True)

def split_data(df):
    years = sorted(df["season_year"].unique())
    test_year = years[-1] if len(years) >= 1 else None
    val_year = years[-2] if len(years) >= 2 else test_year
    train = df[df["season_year"] < val_year].reset_index(drop=True)
    val = df[df["season_year"] == val_year].reset_index(drop=True)
    test = df[df["season_year"] == test_year].reset_index(drop=True)
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

def train_models(df_train, session_type, reg_max_iter=300, reg_learning_rate=0.03,
                 reg_max_depth=3, clf_max_iter=75, clf_learning_rate=0.015,
                 clf_max_depth=2, clf_min_samples_leaf=60, clf_l2_regularization=5.0,
                 random_state=42):
    train_fin = df_train[df_train["is_dnf"] == 0].reset_index(drop=True)
    X_fin, y_fin, _ = get_features_target(train_fin, session_type)
    reg = HistGradientBoostingRegressor(max_iter=reg_max_iter, learning_rate=reg_learning_rate, max_depth=reg_max_depth, random_state=random_state)
    reg.fit(X_fin, y_fin)
    X_dnf, y_dnf = get_dnf_features(df_train)
    clf = HistGradientBoostingClassifier(max_iter=clf_max_iter, learning_rate=clf_learning_rate, max_depth=clf_max_depth, min_samples_leaf=clf_min_samples_leaf, l2_regularization=clf_l2_regularization, random_state=random_state)
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
    full_mode = "--full" in sys.argv
    for session_type in ["RAC", "SPR"]:
        print(f"\ntraining {session_type} model...")
        df = load_dataset(session_type)
        train, val, test = split_data(df)
        years = sorted(df["season_year"].unique())
        reg, clf = train_models(train, session_type)
        std_dev = evaluate(reg, clf, val, session_type, f"Validation {years[-2] if len(years) >= 2 else '?'}")
        evaluate(reg, clf, test, session_type, f"Test {years[-1]}")
        if full_mode:
            print(f"[--full] Retraining {session_type} on 100% of data ({years[0]}–{years[-1]})...")
            reg, clf = train_models(df, session_type)
        save_bundle(reg, clf, std_dev, session_type)