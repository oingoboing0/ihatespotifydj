import requests
from flask import Flask, redirect, request, jsonify, session
import urllib.parse
from datetime import datetime, timezone
import time
import json
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
AUTH_URL = os.getenv("AUTH_URL")
TOKEN_URL = os.getenv("TOKEN_URL")
API_BASE_URL = os.getenv("API_BASE_URL")
access_token = os.getenv("access_token")
refresh_token = os.getenv("refresh_token")
expires_at = os.getenv("expires_at")

# Rate limiting constants
MAX_REQUESTS_PER_SECOND = 10
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 0.1
BATCH_DELAY = 0.1
MAX_IDS_PER_REQUEST = 100
def get_client_token():
    from base64 import b64encode
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    token_url = "https://accounts.spotify.com/api/token"
    
    auth_str = f"{client_id}:{client_secret}"
    headers = {
        "Authorization": "Basic " + b64encode(auth_str.encode()).decode(),
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    
    response = requests.post(token_url, headers=headers, data=data)
    response.raise_for_status()
    return response.json()["access_token"]


client_token = get_client_token()

class RateLimiter:
    def __init__(self, max_requests_per_second):
        self.max_requests_per_second = max_requests_per_second
        self.requests = []
    
    def acquire(self):
        now = time.time()
        # Remove requests older than 1 second
        self.requests = [req_time for req_time in self.requests if now - req_time < 1]
        
        if len(self.requests) >= self.max_requests_per_second:
            # Wait until we can make another request
            sleep_time = 1 - (now - self.requests[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.requests = self.requests[1:]
        
        self.requests.append(time.time())

def fetch_track_features_batch(track_ids, headers, rate_limiter, retry_count=0):
    """Fetch audio features for multiple tracks in a single request"""
    rate_limiter.acquire()
    try:
        # Join track IDs with commas for the query parameter
        ids_param = ','.join(track_ids)
        response = requests.get(f"{API_BASE_URL}/audio-features?ids={ids_param}", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('audio_features', [])
        elif response.status_code == 429:  # Too Many Requests
            if retry_count < MAX_RETRIES:
                retry_after = float(response.headers.get('Retry-After', INITIAL_RETRY_DELAY))
                print(f"Rate limit hit for batch. Retrying after {retry_after:.2f} seconds...")
                time.sleep(retry_after)
                return fetch_track_features_batch(track_ids, headers, rate_limiter, retry_count + 1)
            else:
                print(f"Max retries exceeded for batch")
                return [None] * len(track_ids)
        else:
            print(f"Error fetching features for batch: {response.status_code}")
            print(f"Response content: {response.text}")
            if retry_count < MAX_RETRIES:
                retry_delay = INITIAL_RETRY_DELAY * (1.5 ** retry_count)
                print(f"Retrying in {retry_delay:.2f} seconds...")
                time.sleep(retry_delay)
                return fetch_track_features_batch(track_ids, headers, rate_limiter, retry_count + 1)
            return [None] * len(track_ids)
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {str(e)}")
        if retry_count < MAX_RETRIES:
            retry_delay = INITIAL_RETRY_DELAY * (1.5 ** retry_count)
            print(f"Retrying in {retry_delay:.2f} seconds...")
            time.sleep(retry_delay)
            return fetch_track_features_batch(track_ids, headers, rate_limiter, retry_count + 1)
        return [None] * len(track_ids)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        if retry_count < MAX_RETRIES:
            retry_delay = INITIAL_RETRY_DELAY * (1.5 ** retry_count)
            print(f"Retrying in {retry_delay:.2f} seconds...")
            time.sleep(retry_delay)
            return fetch_track_features_batch(track_ids, headers, rate_limiter, retry_count + 1)
        return [None] * len(track_ids)

def process_all_tracks(track_ids, headers):
    """Process all tracks in batches of 100"""
    rate_limiter = RateLimiter(MAX_REQUESTS_PER_SECOND)
    all_features = []
    
    # Process tracks in batches of MAX_IDS_PER_REQUEST
    for i in range(0, len(track_ids), MAX_IDS_PER_REQUEST):
        batch = track_ids[i:i + MAX_IDS_PER_REQUEST]
        print(f"Processing batch {i//MAX_IDS_PER_REQUEST + 1} of {(len(track_ids) + MAX_IDS_PER_REQUEST - 1)//MAX_IDS_PER_REQUEST}")
        
        # Process this batch
        batch_features = fetch_track_features_batch(batch, headers, rate_limiter)
        all_features.extend(batch_features)
        
        # Small delay between batches
        if i + MAX_IDS_PER_REQUEST < len(track_ids):
            time.sleep(BATCH_DELAY)
    
    return all_features

def process_tracks_in_batches(df):
    """Process tracks in batches and add features to the dataframe"""
    headers = {'Authorization': f"Bearer {client_token}",
    "Content-Type": "application/json"}
    
    # Get all track IDs
    track_ids = df['id'].tolist()
    
    # Process all tracks in batches
    all_features = process_all_tracks(track_ids, headers)
    
    # Create a dictionary mapping track IDs to their features
    features_dict = {f['id']: f for f in all_features if f is not None}
    
    # Add features to the dataframe
    df['audio_features'] = df['id'].map(features_dict)
    
    return df

def main():
    # Load the parquet file
    input_path = r"C:\Users\hedgi\Desktop\spotify_data\processed_data.parquet"
    output_path = r"C:\Users\hedgi\Desktop\spotify_data\processed_for_ml.parquet"
    output_path_csv = r"C:\Users\hedgi\Desktop\spotify_data\processed_for_ml.csv"
    
    print("Loading parquet file...")
    df = pd.read_parquet(input_path)
    
    if not client_token:
        raise ValueError("No client token found. Please ensure authentication is complete.")

    expires_dt = datetime.fromtimestamp(float(expires_at), tz=timezone.utc)
    if expires_dt < datetime.now(timezone.utc):
        raise ValueError("Access token has expired. Please refresh the token.")
    
    print("Processing tracks and fetching features...")
    df_with_features = process_tracks_in_batches(df)
    
    print("Saving enhanced dataset...")
    df_with_features.to_parquet(output_path)
    df_with_features.to_csv(output_path_csv)
    print(f"Enhanced dataset saved to {output_path} and {output_path_csv}")

if __name__ == "__main__":
    main()


