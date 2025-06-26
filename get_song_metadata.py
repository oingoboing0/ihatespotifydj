import requests
from flask import Flask, redirect, request, jsonify, session
import urllib.parse
from datetime import datetime, timezone
import asyncio
from pyspark.sql import SparkSession
import aiohttp
import json
import spark
from pyspark.sql import functions as sf
import os
import re
import pandas as pd
from dotenv import load_dotenv
import random
import time

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
MAX_RETRIES = 3  # Reduced retries since we're being more aggressive
INITIAL_RETRY_DELAY = 0.1  # Reduced initial delay
BATCH_DELAY = 0.1  # Small delay between batches

class RateLimiter:
    def __init__(self, max_requests_per_second):
        self.max_requests_per_second = max_requests_per_second
        self.requests = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            # Remove requests older than 1 second
            self.requests = [req_time for req_time in self.requests if now - req_time < 1]
            
            if len(self.requests) >= self.max_requests_per_second:
                # Wait until we can make another request
                sleep_time = 1 - (now - self.requests[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                self.requests = self.requests[1:]
            
            self.requests.append(time.time())

async def fetch_track_features(session, track_id, headers, rate_limiter, retry_count=0):
    """Fetch audio features for a single track with optimized rate limiting"""
    await rate_limiter.acquire()
    try:
        async with session.get(f"{API_BASE_URL}/audio-features/{track_id}", headers=headers) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 429:  # Too Many Requests
                if retry_count < MAX_RETRIES:
                    # Use the Retry-After header if provided, otherwise use a minimal delay
                    retry_after = float(response.headers.get('Retry-After', INITIAL_RETRY_DELAY))
                    print(f"Rate limit hit for track {track_id}. Retrying after {retry_after:.2f} seconds...")
                    await asyncio.sleep(retry_after)
                    return await fetch_track_features(session, track_id, headers, rate_limiter, retry_count + 1)
                else:
                    print(f"Max retries exceeded for track {track_id}")
                    return None
            else:
                print(f"Error fetching features for track {track_id}: {response.status}")
                return None
    except Exception as e:
        if retry_count < MAX_RETRIES:
            # Minimal delay for other errors
            retry_delay = INITIAL_RETRY_DELAY * (1.5 ** retry_count)
            print(f"Exception fetching features for track {track_id}: {str(e)}. Retrying in {retry_delay:.2f} seconds...")
            await asyncio.sleep(retry_delay)
            return await fetch_track_features(session, track_id, headers, rate_limiter, retry_count + 1)
        else:
            print(f"Max retries exceeded for track {track_id}: {str(e)}")
            return None

async def process_batch(track_ids, headers, rate_limiter):
    """Process a single batch of tracks"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_track_features(session, track_id, headers, rate_limiter) for track_id in track_ids]
        return await asyncio.gather(*tasks)

async def process_all_tracks(track_ids, headers, batch_size=100):
    """Process all tracks in batches using a single event loop"""
    rate_limiter = RateLimiter(MAX_REQUESTS_PER_SECOND)
    all_features = []
    
    for i in range(0, len(track_ids), batch_size):
        batch = track_ids[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} of {(len(track_ids) + batch_size - 1)//batch_size}")
        
        # Process this batch
        batch_features = await process_batch(batch, headers, rate_limiter)
        all_features.extend(batch_features)
        
        # Minimal delay between batches
        if i + batch_size < len(track_ids):
            await asyncio.sleep(BATCH_DELAY)
    
    return all_features

def process_tracks_in_batches(df, batch_size=100):
    """Process tracks in batches and add features to the dataframe"""
    headers = {'Authorization': f"Bearer {access_token}"}
    
    # Get all track IDs
    track_ids = df['id'].tolist()
    
    # Process all tracks using a single event loop
    all_features = asyncio.run(process_all_tracks(track_ids, headers, batch_size))
    
    # Create a dictionary mapping track IDs to their features
    features_dict = {f['id']: f for f in all_features if f is not None}
    
    # Add features to the dataframe
    df['audio_features'] = df['id'].map(features_dict)
    
    return df

def main():
    # Load the parquet file
    input_path = r"C:\Users\hedgi\Desktop\spotify_data\processed_data.parquet"
    output_path = r"C:\Users\hedgi\Desktop\spotify_data\processed_for_ml.parquet"
    
    print("Loading parquet file...")
    df = pd.read_parquet(input_path)
    
    if not access_token:
        raise ValueError("No access token found. Please ensure authentication is complete.")

    expires_dt = datetime.fromtimestamp(float(expires_at), tz=timezone.utc)
    if expires_dt < datetime.now(timezone.utc):
        raise ValueError("Access token has expired. Please refresh the token.")
    
    print("Processing tracks and fetching features...")
    df_with_features = process_tracks_in_batches(df)
    
    print("Saving enhanced dataset...")
    df_with_features.to_parquet(output_path)
    print(f"Enhanced dataset saved to {output_path}")

if __name__ == "__main__":
    main()


