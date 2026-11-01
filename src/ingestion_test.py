import requests
import pandas as pd

base_url = "https://api.motogp.pulselive.com/motogp/v1"

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
    seasons = get_seasons()
    if seasons:
        print("seasons retrieved successfully")
        df_seasons = pd.DataFrame(seasons)
        print(df_seasons)
        events = get_events(seasons[0]['id'])
        if events:
            print("events retrieved successfully")
            df_events = pd.DataFrame(events)
            print(df_events)
            categories = get_categories(events[0]['id'])
            if categories:
                print("categories retrieved successfully")
                df_categories = pd.DataFrame(categories)
                print(df_categories)
                sessions = get_session(events[0]['id'], categories[0]['id'])
                if sessions:
                    print("sessions retrieved successfully")
                    df_sessions = pd.DataFrame(sessions)
                    print(df_sessions)
                    race = next((s for s in sessions if s["type"] == "RAC"), None)
                    if race:
                        session_results = get_session_results(race["id"])
                        if session_results:
                            print("session results retrieved successfully")
                            df_session_results = pd.json_normalize(session_results["classification"])
                            print(df_session_results)
                    else:
                        print("no race session found for this event")