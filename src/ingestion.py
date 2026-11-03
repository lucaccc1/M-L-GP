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
        response = requests.get(f"{url}/{endpoint}", params=params, timeout=15)  # im just sending a get request to the url
        response.raise_for_status()  # this will raise an error if the request was unsuccessful
        data = response.json()
        return data
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        print(f"body: {e.response.text}")  
        return None
    except requests.exceptions.RequestException as e:
        print(f"an error occurred: {e}")
        return None

def get_seasons():
    return my_request(base_url, f"results/seasons")

def get_events(season_id):
    return my_request(base_url, f"results/events", params={"seasonUuid": season_id})

def get_categories(event_id):
    return my_request(base_url, f"results/categories", params={"eventUuid": event_id})

def get_session(event_id, category_id):
    return my_request(base_url, f"results/sessions", params={"eventUuid": event_id,
                                                            "categoryUuid": category_id})

def get_session_results(session_id):
    return my_request(base_url, f"results/session/{session_id}/classification")

if __name__ == "__main__":
    if os.path.exists(seasons_path):
        existing_seasons = pd.read_csv(seasons_path, low_memory=False)
    else:
        existing_seasons = pd.DataFrame()

    if os.path.exists(events_path):
        existing_events = pd.read_csv(events_path, low_memory=False)
    else:
        existing_events = pd.DataFrame()

    if os.path.exists(sessions_path):
        existing_sessions = pd.read_csv(sessions_path, low_memory=False)
    else:
        existing_sessions = pd.DataFrame()

    if os.path.exists(result_path):
        existing_results = pd.read_csv(result_path, low_memory=False)
    else:
        existing_results = pd.DataFrame()
    existing_event_ids = set(existing_events['id'].dropna()) if not existing_events.empty and 'id' in existing_events else set()
    existing_session_ids = set(existing_sessions['id'].dropna()) if not existing_sessions.empty and 'id' in existing_sessions else set()
    finished_event_ids = set(existing_events[existing_events['status'] == 'FINISHED']['id'].dropna()) if not existing_events.empty and 'status' in existing_events else set()
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
                    if event_id not in existing_event_ids:
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
                                    if session_id not in existing_session_ids:
                                        all_sessions.append(session)
                                    if session_id not in existing_session_ids and session_status == 'FINISHED':
                                        session_results = get_session_results(session_id)
                                        if session_results and "classification" in session_results:
                                            print("session results retrieved successfully")
                                            classification = session_results["classification"]
                                            for result in classification:
                                                result['season_year'] = season_year
                                                result['event_id'] = event_id
                                                result['session_id'] = session_id
                                                result['session_type'] = session_type
                                                all_results.append(result)
    os.makedirs("data/raw", exist_ok=True)
    df_seasons.to_csv(seasons_path, index=False)
    if all_events:
        df_new_events = pd.DataFrame(all_events)
        df_events = pd.concat([existing_events, df_new_events], ignore_index=True).drop_duplicates(subset=['id'])
        df_events.to_csv(events_path, index=False)
    if all_sessions:
        df_new_sessions = pd.DataFrame(all_sessions)
        df_sessions = pd.concat([existing_sessions, df_new_sessions], ignore_index=True).drop_duplicates(subset=['id'])
        df_sessions.to_csv(sessions_path, index=False)
    if all_results:
        df_new_results = pd.json_normalize(all_results)
        df_results = pd.concat([existing_results, df_new_results], ignore_index=True)
        df_results.to_csv(result_path, index=False)
        print("new session results saved successfully")
    else:
        print("no new session results to save")