import pandas as pd
import numpy as np
import joblib

def load_bundle(session_type="RAC"):
    return joblib.load(f"models/model_{session_type.lower()}.pkl")

def get_latest_session(session_type="RAC"):
    df = pd.read_csv("data/processed/dataset_ml.csv", low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    sub = df[df["session_type"] == session_type].sort_values(by=["date", "session_id"]).reset_index(drop=True)
    last_session_id = sub["session_id"].iloc[-1]
    return sub[sub["session_id"] == last_session_id].reset_index(drop=True)

def build_upcoming_race(grid_df, circuit_name, session_type="RAC", is_wet=0):
    df = pd.read_csv("data/processed/dataset_ml.csv", low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["date", "session_id"]).reset_index(drop=True)
    rows = []
    for _, r in grid_df.iterrows():
        rider, const, grid = r["rider_name"], r["constructor_name"], r["grid_position"]
        r_hist = df[df["rider_name"] == rider]
        r_rac = r_hist[r_hist["session_type"] == "RAC"]
        r_spr = r_hist[r_hist["session_type"] == "SPR"]
        r_circ = r_hist[r_hist["circuit_name"] == circuit_name]
        c_hist = df[df["constructor_name"] == const].groupby("session_id", sort=False)["position_cleaned"].mean()
        c_circ = df[(df["constructor_name"] == const) & (df["circuit_name"] == circuit_name)].groupby("session_id", sort=False)["position_cleaned"].mean()
        rows.append({
            "rider_name": rider,
            "constructor_name": const,
            "grid_position": grid,
            "is_front_row": 1 if grid <= 3 else 0,
            "is_wet": is_wet,
            "rider_last3_avg": r_hist["position_cleaned"].tail(3).mean() if len(r_hist) else np.nan,
            "rider_last5_avg": r_hist["position_cleaned"].tail(5).mean() if len(r_hist) else np.nan,
            "rider_last3_avg_rac": r_rac["position_cleaned"].tail(3).mean() if len(r_rac) else np.nan,
            "rider_last3_avg_spr": r_spr["position_cleaned"].tail(3).mean() if len(r_spr) else np.nan,
            "dnf_ratio_last5": r_hist["is_dnf"].tail(5).mean() if len(r_hist) else np.nan,
            "dnf_ratio_last10": r_hist["is_dnf"].tail(10).mean() if len(r_hist) else np.nan,
            "rider_avg_gain": r_hist["grid_delta"].tail(5).mean() if len(r_hist) else np.nan,
            "rider_circuit_avg": r_circ["position_cleaned"].mean() if len(r_circ) else np.nan,
            "constructor_circuit_avg": c_circ.mean() if len(c_circ) else np.nan,
            "constructor_last3_avg": c_hist.tail(3).mean() if len(c_hist) else np.nan,
            "constructor_last5_avg": c_hist.tail(5).mean() if len(c_hist) else np.nan,
            "sprint_pos_same_weekend": r.get("sprint_pos_same_weekend", np.nan) if session_type == "RAC" else np.nan
        })
    return pd.DataFrame(rows)

def predict_session(df_session, bundle):
    reg, clf = bundle["reg"], bundle["clf"]
    cols = list(reg.feature_names_in_)
    df_out = df_session.copy()
    df_out["pred_pos"] = reg.predict(df_out[cols])
    df_out["dnf_prob"] = clf.predict_proba(df_out[cols])[:, 1]
    return df_out

def simulate_race(df_pred, std_dev, n_sims=100000):
    rng = np.random.default_rng(42)
    n_riders = len(df_pred)
    mu = df_pred["pred_pos"].values
    p_dnf = df_pred["dnf_prob"].values
    scores = rng.normal(loc=mu, scale=std_dev, size=(n_sims, n_riders))
    crashes = rng.random((n_sims, n_riders)) < p_dnf
    scores = np.where(crashes, 999.0, scores)
    ranks = np.argsort(np.argsort(scores, axis=1), axis=1)
    keep_cols = [c for c in ["rider_name", "constructor_name", "grid_position", "position_cleaned", "pred_pos", "dnf_prob"] if c in df_pred.columns]
    res = df_pred[keep_cols].copy()
    res["win_pct"] = np.round((ranks == 0).mean(axis=0) * 100, 2)
    res["podium_pct"] = np.round((ranks < 3).mean(axis=0) * 100, 2)
    res["top10_pct"] = np.round((ranks < 10).mean(axis=0) * 100, 2)
    return res.sort_values("win_pct", ascending=False).reset_index(drop=True)

def get_race_info(df_race):
    circuit = df_race["circuit_name"].iloc[0] if "circuit_name" in df_race.columns and pd.notna(df_race["circuit_name"].iloc[0]) else df_race["event_id"].iloc[0]
    race_date = str(df_race["date"].iloc[0])[:10] if "date" in df_race.columns else "N/A"
    return f"{circuit} ({race_date})"

if __name__ == "__main__":
    for session_type in ["SPR", "RAC"]:
        bundle = load_bundle(session_type)
        df_race = get_latest_session(session_type)
        df_pred = predict_session(df_race, bundle)
        results = simulate_race(df_pred, bundle["std_dev"])
        print(f"\n{session_type} - {get_race_info(df_race)} simulation (100.000 sims)")
        print(results.to_string(index=False))