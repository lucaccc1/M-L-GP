import os
import requests
import pandas as pd

base_url = "https://api.motogp.pulselive.com/motogp/v1"
all_events = []
all_sessions = []
all_results = []

def my_request(url, endpoint="", params=None):
    try:
        response = requests.get(f"{url}/{endpoint}", params=params, timeout=15)  
        response.raise_for_status()  
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
    seasons = get_seasons()
    if seasons:
        print("seasons retrieved successfully")
        for season in seasons:
            season_id = season['id']
            season_year = season['year']
            events = get_events(season_id)
            if not events:
                continue
            for event in events:
                event_id = event['id']
                all_events.append(event)
                categories = get_categories(event_id)
                if not categories:
                    continue
                for category in categories:
                    if "MotoGP" not in category.get("name", ""):
                        continue
                    category_id = category['id']
                    sessions = get_session(event_id, category_id)
                    if not sessions:
                        continue
                    for session in sessions:
                        session_id = session['id']
                        all_sessions.append(session)
                        session_results = get_session_results(session_id)
                        if not session_results:
                            continue
                        classification = session_results.get("classification", [])
                        for result in classification:
                            result['season_year'] = season_year
                            result['event_id'] = event_id
                            result['session_id'] = session_id
                            all_results.append(result)

        os.makedirs("data/raw", exist_ok=True)
        pd.DataFrame(seasons).to_csv("data/raw/seasons.csv", index=False)
        pd.DataFrame(all_events).to_csv("data/raw/events.csv", index=False)
        pd.DataFrame(all_sessions).to_csv("data/raw/sessions.csv", index=False)
        pd.json_normalize(all_results).to_csv("data/raw/results.csv", index=False)
        print("download completed and files saved in data/raw/")