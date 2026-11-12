import os
import requests
import pandas as pd

base_url = "https://api.motogp.pulselive.com/motogp/v1"
events_path = "data/raw/events.csv"
result_path = "data/raw/results.csv"
seasons_path = "data/raw/seasons.csv"
sessions_path = "data/raw/sessions.csv"

def my_request(url, endpoint="", params=None):
    try:
        response = requests.get(f"{url}/{endpoint}", params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        print(f"body: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"an error occurred: {e}")
        return None

def get_seasons():
    return my_request(base_url, "results/seasons")

def get_events(season_id):
    return my_request(base_url, "results/events", params={"seasonUuid": season_id})

def get_categories(event_id):
    return my_request(base_url, "results/categories", params={"eventUuid": event_id})

def get_session(event_id, category_id):
    return my_request(base_url, "results/sessions", params={"eventUuid": event_id, "categoryUuid": category_id})

def get_session_results(session_id):
    return my_request(base_url, f"results/session/{session_id}/classification")

if __name__ == "__main__":
    existing_seasons = pd.read_csv(seasons_path, low_memory=False) if os.path.exists(seasons_path) else pd.DataFrame()
    existing_events = pd.read_csv(events_path, low_memory=False) if os.path.exists(events_path) else pd.DataFrame()
    existing_sessions = pd.read_csv(sessions_path, low_memory=False) if os.path.exists(sessions_path) else pd.DataFrame()
    existing_results = pd.read_csv(result_path, low_memory=False) if os.path.exists(result_path) else pd.DataFrame()

    finished_event_ids = set(existing_events[existing_events['status'] == 'FINISHED']['id'].dropna()) if not existing_events.empty and 'status' in existing_events else set()
    existing_result_session_ids = set(existing_results['session_id'].dropna()) if not existing_results.empty and 'session_id' in existing_results else set()

    all_events = []
    all_sessions = []
    all_results = []
    seasons = get_seasons()

    if seasons:
        print("seasons retrieved successfully")
        df_seasons = pd.DataFrame(seasons)
        for season in seasons:
            season_id = season['id']
            season_year = season['year']
            events = get_events(season_id)
            if events:
                for event in events:
                    event_id = event['id']
                    event_status = event.get('status')
                    if event_id in finished_event_ids and event_status == 'FINISHED':
                        continue
                    all_events.append(event)
                    categories = get_categories(event_id)
                    if categories:
                        for category in categories:
                            if "MotoGP" not in category.get("name", ""):
                                continue
                            category_id = category['id']
                            sessions = get_session(event_id, category_id)
                            if sessions:
                                for session in sessions:
                                    session_id = session['id']
                                    session_type = session.get('type')
                                    session_status = session.get('status')
                                    if session_id not in existing_result_session_ids:
                                        all_sessions.append(session)
                                        if session_status == 'FINISHED':
                                            session_results = get_session_results(session_id)
                                            if session_results and "classification" in session_results:
                                                print(f"session results retrieved: {session_type} ({session_id})")
                                                for result in session_results["classification"]:
                                                    result['season_year'] = season_year
                                                    result['event_id'] = event_id
                                                    result['session_id'] = session_id
                                                    result['session_type'] = session_type
                                                    all_results.append(result)

    os.makedirs("data/raw", exist_ok=True)
    df_seasons.to_csv(seasons_path, index=False)
    if all_events:
        df_events = pd.concat([existing_events, pd.DataFrame(all_events)], ignore_index=True).drop_duplicates(subset=['id'], keep='last')
        df_events.to_csv(events_path, index=False)
    if all_sessions:
        df_sessions = pd.concat([existing_sessions, pd.DataFrame(all_sessions)], ignore_index=True).drop_duplicates(subset=['id'], keep='last')
        df_sessions.to_csv(sessions_path, index=False)
    if all_results:
        df_results = pd.concat([existing_results, pd.json_normalize(all_results)], ignore_index=True)
        df_results.to_csv(result_path, index=False)
        print("new session results saved successfully")
    else:
        print("no new session results to save")