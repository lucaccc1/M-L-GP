import pandas as pd
import numpy as np
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

def clean_columns(df):
    df_cleaned = df.rename(columns={
        'type': 'session_type',
        'rider.id':'rider_id',
        'rider.full_name':'rider_name',
        'constructor.name':'constructor_name',
        'team.name':'team_name'
    }).copy()
    df_cleaned['is_dnf'] = np.where(df_cleaned['status'] != 'INSTND', 1, 0)
    df_cleaned['position_cleaned'] = df_cleaned['position'].fillna(25).astype(int)
    return df_cleaned

if __name__ == "__main__":
    results, sessions = load_data()
    filtered_sessions = filter_sessions_year(results, sessions)
    cleaned_sessions = clean_columns(filtered_sessions)
    os.makedirs("data/processed", exist_ok=True)
    cleaned_sessions.to_csv("data/processed/cleaned_sessions.csv", index=False)
