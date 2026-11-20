import pandas as pd
import numpy as np
import os

def load_data():
    df_races = pd.read_csv("data/processed/cleaned_sessions.csv", low_memory=False)
    df_sessions = pd.read_csv("data/raw/sessions.csv", low_memory=False).drop_duplicates(subset=["id"])
    df_sessions["circuit_name"] = df_sessions["circuit"].astype(str).str.extract(r"['\"]name['\"]:\s*(['\"])(.*?)\1")[1].fillna(df_sessions["circuit"])
    cols_to_merge = [c for c in ["id", "date", "condition", "circuit_name"] if c == "id" or c not in df_races.columns]
    df_races = df_races.merge(df_sessions[cols_to_merge], left_on="session_id", right_on="id", how="left").drop(columns=["id"], errors="ignore")
    return df_races

def order_by_date(df):
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(by=["date", "session_id", "grid_position"]).reset_index(drop=True)

def grid_delta(df):
    df["grid_delta"] = df["grid_position"] - df["position_cleaned"]
    df["is_front_row"] = np.where(df["grid_position"] <= 3, 1, 0)
    return df

def rider_last3_avg(df):
    df["rider_last3_avg"] = df.groupby("rider_id")["position_cleaned"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    return df

def rider_last5_avg(df):
    df["rider_last5_avg"] = df.groupby("rider_id")["position_cleaned"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    return df

def rider_session_avg(df):
    rac = df[df["session_type"] == "RAC"].copy()
    rac["rider_last3_avg_rac"] = rac.groupby("rider_id")["position_cleaned"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    spr = df[df["session_type"] == "SPR"].copy()
    spr["rider_last3_avg_spr"] = spr.groupby("rider_id")["position_cleaned"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df = df.merge(rac[["session_id", "rider_id", "rider_last3_avg_rac"]], on=["session_id", "rider_id"], how="left")
    df = df.merge(spr[["session_id", "rider_id", "rider_last3_avg_spr"]], on=["session_id", "rider_id"], how="left")
    return df

def sprint_same_weekend(df):
    spr = df[df["session_type"] == "SPR"][["season_year", "event_id", "rider_id", "position_cleaned"]].drop_duplicates(subset=["season_year", "event_id", "rider_id"], keep="last").copy()
    spr = spr.rename(columns={"position_cleaned": "sprint_pos_same_weekend"})
    df = df.merge(spr, on=["season_year", "event_id", "rider_id"], how="left")
    df["sprint_pos_same_weekend"] = np.where(df["session_type"] == "RAC", df["sprint_pos_same_weekend"], np.nan)
    return df

def weather(df):
    df["is_wet"] = np.where(df["condition"].str.contains("Wet|Rain", case=False, na=False), 1, 0)
    return df

def affidability(df):
    df["dnf_ratio_last5"] = df.groupby("rider_id")["is_dnf"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df["dnf_ratio_last10"] = df.groupby("rider_id")["is_dnf"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    return df

def grid_delta_gain(df):
    df["rider_avg_gain"] = df.groupby("rider_id")["grid_delta"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    return df

def circuit_affinity(df):
    df["rider_circuit_avg"] = df.groupby(["rider_id", "circuit_name"])["position_cleaned"].transform(lambda x: x.shift(1).expanding().mean())
    c_circ = df.groupby(["session_id", "circuit_name", "constructor_name", "date"])["position_cleaned"].mean().reset_index()
    c_circ = c_circ.sort_values(by=["date", "session_id"]).reset_index(drop=True)
    c_circ["constructor_circuit_avg"] = c_circ.groupby(["constructor_name", "circuit_name"])["position_cleaned"].transform(lambda x: x.shift(1).expanding().mean())
    return df.merge(c_circ[["session_id", "constructor_name", "constructor_circuit_avg"]], on=["session_id", "constructor_name"], how="left")

def constructor_rolling_avg(df):
    c_avg = df.groupby(["session_id", "constructor_name", "date"])["position_cleaned"].mean().reset_index()
    c_avg = c_avg.sort_values(by=["date", "session_id"]).reset_index(drop=True)
    c_avg["constructor_last3_avg"] = c_avg.groupby("constructor_name")["position_cleaned"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    c_avg["constructor_last5_avg"] = c_avg.groupby("constructor_name")["position_cleaned"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    return df.merge(c_avg[["session_id", "constructor_name", "constructor_last3_avg", "constructor_last5_avg"]], on=["session_id", "constructor_name"], how="left")

def circuit_dnf_rate(df):
    c_dnf = df.groupby(["session_id", "circuit_name", "session_type", "date"], as_index=False)["is_dnf"].mean()
    c_dnf = c_dnf.rename(columns={"is_dnf": "session_dnf_rate"})
    c_dnf = c_dnf.sort_values(by=["date", "session_id"]).reset_index(drop=True)
    c_dnf["circuit_dnf_rate"] = c_dnf.groupby(["circuit_name", "session_type"])["session_dnf_rate"].transform(lambda x: x.shift(1).expanding().mean())
    c_dnf = df.merge(c_dnf[["session_id", "circuit_dnf_rate"]], on="session_id", how="left")
    return c_dnf

def clean_finish_streak(df):
    is_clean = (df["is_dnf"] == 0).astype(int)
    dnf_blocks = df["is_dnf"].eq(1).groupby(df["rider_id"]).cumsum()
    raw_streak = is_clean.groupby([df["rider_id"], dnf_blocks]).cumsum()
    df["clean_finish_streak"] = (raw_streak.groupby(df["rider_id"]).shift(1).fillna(0).astype(int))
    return df

if __name__ == "__main__":
    df = load_data()
    df = order_by_date(df)
    df = grid_delta(df)
    df = rider_last3_avg(df)
    df = rider_last5_avg(df)
    df = rider_session_avg(df)
    df = sprint_same_weekend(df)
    df = weather(df)
    df = affidability(df)
    df = grid_delta_gain(df)
    df = circuit_affinity(df)
    df = constructor_rolling_avg(df)
    df = circuit_dnf_rate(df)
    df = clean_finish_streak(df)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/dataset_ml.csv", index=False)