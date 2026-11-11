import pandas as pd
import numpy as np
import os

def load_data():
  df_races = pd.read_csv("data/processed/cleaned_sessions.csv", low_memory=False)
  df_sessions = pd.read_csv("data/raw/sessions.csv", low_memory=False)
  if "date" not in df_races.columns:
    df_races = df_races.merge(df_sessions[["id", "date"]], left_on="session_id", right_on="id", how="left")
  return df_races

def order_by_date(df):
  df["date"] = pd.to_datetime(df["date"])
  return df.sort_values(by="date").reset_index(drop=True)

def grid_delta(df):
  df["grid_delta"] = df["grid_position"] - df["position_cleaned"]
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
  df = df.merge(rac[["session_id", "rider_id", "rider_last3_avg_rac"]],on=["session_id", "rider_id"],how="left",)
  df = df.merge(spr[["session_id", "rider_id", "rider_last3_avg_spr"]],on=["session_id", "rider_id"],how="left",)
  return df


def constructor_last3_avg(df):
  c_avg = (df.groupby(["event_id", "session_type", "constructor_name", "date"])["position_cleaned"].mean().reset_index())
  c_avg = c_avg.sort_values("date").reset_index(drop=True)
  c_avg["constructor_last3_avg"] = c_avg.groupby("constructor_name")["position_cleaned"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
  df = df.merge(c_avg[["event_id", "session_type", "constructor_name", "constructor_last3_avg",]], on=["event_id", "session_type", "constructor_name"], how="left",)
  return df

def sprint_same_weekend(df):
  spr = df[df["session_type"] == "SPR"][["season_year", "event_id", "rider_id", "position_cleaned"]].copy()
  spr = spr.rename(columns={"position_cleaned": "sprint_pos_same_weekend"})
  df = df.merge(spr, on=["season_year", "event_id", "rider_id"], how="left")
  return df

if __name__ == "__main__":
  df = load_data()
  df = order_by_date(df)
  df = grid_delta(df)
  df = rider_last3_avg(df)
  df = rider_last5_avg(df)
  df = rider_session_avg(df)
  df = constructor_last3_avg(df)
  df = sprint_same_weekend(df)
  os.makedirs("data/processed", exist_ok=True)
  df.to_csv("data/processed/dataset_ml.csv", index=False)