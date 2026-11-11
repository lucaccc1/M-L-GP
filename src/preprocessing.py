import pandas as pd
import numpy as np
import os

def load_data():
    results = pd.read_csv("data/raw/results.csv", low_memory=False)
    sessions = pd.read_csv("data/raw/sessions.csv", low_memory=False)
    return results, sessions

def filter_sessions_year(results, sessions, start_year=2016):
    valid_types = ['Q', 'SPR', 'RAC', 'QP', 'Q2']
    df_merged = pd.merge(results, sessions[['id', 'type', 'number']], left_on='session_id', right_on='id', how='left')
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

def starting_grid(df):
    qualifying = df[df['session_type'].isin(['Q', 'QP', 'Q2'])].copy()
    q2 = qualifying[qualifying['number'] == 2][['season_year', 'event_id', 'rider_id', 'position_cleaned']].copy()
    q2 = q2.rename(columns={'position_cleaned': 'grid_position'})
    q1 = qualifying[qualifying['number'] == 1][['season_year', 'event_id', 'rider_id', 'position_cleaned']].copy()
    q1_not_in_q2 = q1[q1['position_cleaned'] >= 3][['season_year', 'event_id', 'rider_id', 'position_cleaned']].copy()
    q1_not_in_q2 = q1_not_in_q2.rename(columns={'position_cleaned': 'grid_position'})
    q1_not_in_q2['grid_position'] = q1_not_in_q2['grid_position'] + 10
    grid = pd.concat([q2, q1_not_in_q2[['season_year', 'event_id', 'rider_id', 'grid_position']]], ignore_index=True)
    races = df[df['session_type'].isin(['RAC', 'SPR'])].copy()
    races_with_grid = pd.merge(races, grid, on=['season_year', 'event_id', 'rider_id'], how='left')
    races_with_grid['grid_position'] = races_with_grid['grid_position'].fillna(22).astype(int)
    return races_with_grid

def parse_time(time_str):
    if pd.isna(time_str) or not isinstance(time_str, str):
        return np.nan
    if ':' in time_str:
        minutes, seconds = time_str.split(':')
        total_seconds = float(minutes) * 60 + float(seconds)
    else:
        total_seconds = float(time_str)
    return total_seconds

if __name__ == "__main__":
    results, sessions = load_data()
    filtered_sessions = filter_sessions_year(results, sessions)
    cleaned_sessions = clean_columns(filtered_sessions)
    final_races_df = starting_grid(cleaned_sessions)
    final_races_df['time_seconds'] = final_races_df['time'].apply(parse_time)
    os.makedirs("data/processed", exist_ok=True)
    final_races_df.to_csv("data/processed/cleaned_sessions.csv", index=False)
