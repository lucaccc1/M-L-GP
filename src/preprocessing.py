import pandas as pd
import os

def load_data():
    results = pd.read_csv("data/raw/results.csv", low_memory=False)
    sessions = pd.read_csv("data/raw/sessions.csv", low_memory=False)
    return results, sessions

def filter_sessions_year(results, sessions, start_year=2016):
    valid_types = ['Q', 'SPR', 'RAC', 'QP', 'Q2']
    df_merged = pd.merge(results, sessions[['id', 'type']], left_on='session_id', right_on='id', how='left')
    df_filtered = df_merged[(df_merged['season_year'] >= start_year) & (df_merged['type'].isin(valid_types))].copy()
    return df_filtered

if __name__ == "__main__":
    results, sessions = load_data()
    filtered_sessions = filter_sessions_year(results, sessions)
    os.makedirs("data/processed", exist_ok=True)
    filtered_sessions.to_csv("data/processed/filtered_sessions.csv", index=False)
