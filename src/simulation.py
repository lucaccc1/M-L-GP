import ast
import pandas as pd
import numpy as np
import joblib

def load_bundle(session_type="RAC"):
    return joblib.load(f"models/model_{session_type.lower()}.pkl")

def extract_circuit_name(event_row):
    if "circuit.name" in event_row and pd.notna(event_row["circuit.name"]):
        return str(event_row["circuit.name"])
    if "circuit" in event_row and pd.notna(event_row["circuit"]):
        val = event_row["circuit"]
        if isinstance(val, dict):
            return str(val.get("name", event_row.get("name", event_row["id"])))
        if isinstance(val, str) and val.startswith("{"):
            try:
                parsed = ast.literal_eval(val)
                return str(parsed.get("name", event_row.get("name", event_row["id"])))
            except Exception:
                pass
    return str(event_row.get("name", event_row["id"]))

def estimate_grid_from_history(roster_df, df_ml):
    est_grids = []
    for _, r in roster_df.iterrows():
        r_hist = df_ml[df_ml["rider_id"] == r["rider_id"]] if "rider_id" in r and pd.notna(r["rider_id"]) else df_ml[df_ml["rider_name"] == r["rider_name"]]
        r_events = r_hist[["event_id", "grid_position"]].drop_duplicates()
        avg_grid = r_events["grid_position"].tail(3).mean() if len(r_events) else 15.0
        est_grids.append(avg_grid)
    out = roster_df.copy()
    out["grid_position"] = pd.Series(est_grids).rank(method="first").astype(int)
    return out

def extract_real_grid(ev_results, sessions):
    ev_res = pd.merge(ev_results, sessions[["id", "type", "number"]], left_on="session_id", right_on="id", how="left")
    ev_res["pos_clean"] = ev_res["position"].fillna(25).astype(int)
    qual = ev_res[ev_res["type"].isin(["Q", "QP", "Q2"])].copy()
    q2 = qual[qual["number"] == 2][["rider.id", "rider.full_name", "constructor.name", "pos_clean"]].rename(columns={"pos_clean": "grid_position"})
    q1 = qual[qual["number"] == 1][["rider.id", "rider.full_name", "constructor.name", "pos_clean"]].copy()
    q1_out = q1[q1["pos_clean"] >= 3].copy()
    q1_out["grid_position"] = q1_out["pos_clean"] + 10
    grid = pd.concat([q2, q1_out[["rider.id", "rider.full_name", "constructor.name", "grid_position"]]], ignore_index=True).drop_duplicates(subset=["rider.id"])
    all_riders = ev_res[["rider.id", "rider.full_name", "constructor.name"]].drop_duplicates(subset=["rider.id"])
    grid = pd.merge(all_riders, grid[["rider.id", "grid_position"]], on="rider.id", how="left")
    grid["grid_position"] = grid["grid_position"].fillna(22).astype(int)
    return grid.rename(columns={"rider.id": "rider_id", "rider.full_name": "rider_name", "constructor.name": "constructor_name"})

def detect_weekend_state(df_ml):
    events = pd.read_csv("data/raw/events.csv", low_memory=False)
    sessions = pd.read_csv("data/raw/sessions.csv", low_memory=False).drop_duplicates(subset=["id"], keep="last")
    results = pd.read_csv("data/raw/results.csv", low_memory=False)
    if "test" in events.columns:
        events = events[events["test"] != True].copy()
    sort_col = "date_start" if "date_start" in events.columns else "id"
    events = events.sort_values(by=sort_col).reset_index(drop=True)
    upcoming = events[events["status"] != "FINISHED"]
    next_event = upcoming.iloc[0]
    event_id = next_event["id"]
    circuit_name = extract_circuit_name(next_event)
    race_date = str(next_event.get("date_start", "N/A"))[:10]
    ev_results = results[results["event_id"] == event_id].copy()
    finished_types = set(ev_results["session_type"].dropna()) if not ev_results.empty else set()
    has_qual = bool(finished_types & {"Q", "QP", "Q2"})
    has_spr = "SPR" in finished_types
    is_wet = 0
    if "condition" in sessions.columns and "event_id" in sessions.columns:
        ev_conds = sessions[sessions["event_id"] == event_id]["condition"].dropna().astype(str)
        if any("wet" in c.lower() for c in ev_conds):
            is_wet = 1
    if not has_qual:
        state = "PRE_QUALIFYING"
        last_rac_sid = df_ml[df_ml["session_type"] == "RAC"]["session_id"].iloc[-1]
        roster = df_ml[df_ml["session_id"] == last_rac_sid][["rider_id", "rider_name", "constructor_name"]].drop_duplicates().reset_index(drop=True)
        grid_df = estimate_grid_from_history(roster, df_ml)
        sessions_to_predict = ["SPR", "RAC"]
    elif not has_spr:
        state = "POST_QUALIFYING"
        grid_df = extract_real_grid(ev_results, sessions)
        sessions_to_predict = ["SPR", "RAC"]
    else:
        state = "POST_SPRINT"
        grid_df = extract_real_grid(ev_results, sessions)
        spr_res = ev_results[ev_results["session_type"] == "SPR"][["rider.id", "position"]].copy()
        spr_res["sprint_pos_same_weekend"] = spr_res["position"].fillna(25).astype(int)
        spr_res = spr_res.rename(columns={"rider.id": "rider_id"}).drop_duplicates(subset=["rider_id"])
        grid_df = pd.merge(grid_df, spr_res[["rider_id", "sprint_pos_same_weekend"]], on="rider_id", how="left")
        sessions_to_predict = ["RAC"]
    return {
        "state": state,
        "circuit_name": circuit_name,
        "race_date": race_date,
        "is_wet": is_wet,
        "grid_df": grid_df,
        "sessions_to_predict": sessions_to_predict
    }

def trailing_clean_streak(dnf_series):
    streak = 0
    for val in reversed(dnf_series.tolist()):
        if int(val) == 0:
            streak += 1
        else:
            break
    return streak

def build_upcoming_race(df_ml, grid_df, circuit_name, session_type="RAC", is_wet=0):
    circ_sess = df_ml[df_ml["circuit_name"].str.contains(circuit_name, case=False, regex=False, na=False) & (df_ml["session_type"] == session_type)]
    circ_dnf_val = (circ_sess.groupby("session_id", sort=False)["is_dnf"].mean().mean() if len(circ_sess) else np.nan)
    rows = []
    for _, r in grid_df.iterrows():
        rider, const, grid = r["rider_name"], r["constructor_name"], r["grid_position"]
        r_hist = df_ml[df_ml["rider_id"] == r["rider_id"]] if "rider_id" in r and pd.notna(r["rider_id"]) else df_ml[df_ml["rider_name"] == rider]
        r_rac = r_hist[r_hist["session_type"] == "RAC"]
        r_spr = r_hist[r_hist["session_type"] == "SPR"]
        r_circ = r_hist[r_hist["circuit_name"].str.contains(circuit_name, case=False, regex=False, na=False)]
        c_hist = df_ml[df_ml["constructor_name"] == const].groupby("session_id", sort=False)["position_cleaned"].mean()
        c_circ = df_ml[(df_ml["constructor_name"] == const) & (df_ml["circuit_name"].str.contains(circuit_name, case=False, regex=False, na=False))].groupby("session_id", sort=False)["position_cleaned"].mean()
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
            "circuit_dnf_rate": circ_dnf_val,
            "clean_finish_streak": trailing_clean_streak(r_hist["is_dnf"]) if len(r_hist) else 0,
            "sprint_pos_same_weekend": r.get("sprint_pos_same_weekend", np.nan) if session_type == "RAC" else np.nan
        })
    return pd.DataFrame(rows)

def predict_session(df_session, bundle):
    reg, clf = bundle["reg"], bundle["clf"]
    cols = list(reg.feature_names_in_)
    df_out = df_session.copy()
    df_out["pred_pos"] = np.clip(reg.predict(df_out[cols]), 1.0, 22.0)
    df_out["dnf_prob"] = clf.predict_proba(df_out[cols])[:, 1]
    return df_out

def simulate_race(df_pred, std_dev, n_sims=10000):
    rng = np.random.default_rng(42)
    n_riders = len(df_pred)
    mu = df_pred["pred_pos"].values
    p_dnf = df_pred["dnf_prob"].values
    scores = rng.normal(loc=mu, scale=std_dev, size=(n_sims, n_riders))
    crashes = rng.random((n_sims, n_riders)) < p_dnf
    scores = np.where(crashes, 999.0, scores)
    ranks = np.argsort(np.argsort(scores, axis=1), axis=1)
    keep_cols = [c for c in ["rider_name", "constructor_name", "grid_position", "sprint_pos_same_weekend", "pred_pos", "dnf_prob", "position_cleaned"] if c in df_pred.columns]
    res = df_pred[keep_cols].copy()
    res["win_pct"] = np.round((ranks == 0).mean(axis=0) * 100, 2)
    res["podium_pct"] = np.round((ranks < 3).mean(axis=0) * 100, 2)
    res["top10_pct"] = np.round((ranks < 10).mean(axis=0) * 100, 2)
    return res.sort_values("win_pct", ascending=False).reset_index(drop=True)

def get_completed_event(df_ml, event_id=None):
    if event_id is None:
        event_id = df_ml["event_id"].iloc[-1]
    ev_df = df_ml[df_ml["event_id"] == event_id].copy()
    circuit_name = ev_df["circuit_name"].iloc[0]
    race_date = str(ev_df["date"].max())[:10]
    sessions_data = {}
    for st_type in ["SPR", "RAC"]:
        sub = ev_df[ev_df["session_type"] == st_type].reset_index(drop=True)
        if not sub.empty:
            sessions_data[st_type] = sub
    return {
        "state": "BACKTEST",
        "circuit_name": circuit_name,
        "race_date": race_date,
        "sessions_data": sessions_data
    }

if __name__ == "__main__":
    df_ml = pd.read_csv("data/processed/dataset_ml.csv", low_memory=False)
    df_ml["date"] = pd.to_datetime(df_ml["date"])
    df_ml = df_ml.sort_values(by=["date", "session_id"]).reset_index(drop=True)
    weekend = detect_weekend_state(df_ml)
    print(f"Prossimo GP: {weekend['circuit_name']} ({weekend['race_date']}) | Stato rilevato: {weekend['state']}")
    for session_type in weekend["sessions_to_predict"]:
        bundle = load_bundle(session_type)
        df_upcoming = build_upcoming_race(df_ml, weekend["grid_df"], weekend["circuit_name"], session_type, weekend["is_wet"])
        df_pred = predict_session(df_upcoming, bundle)
        results = simulate_race(df_pred, bundle["std_dev"])
        print(f"\n=== Previsione {session_type} - {weekend['circuit_name']} [{weekend['state']}] ===")
        print(results.to_string(index=False))