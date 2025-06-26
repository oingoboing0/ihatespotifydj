import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_analyze_data(folder_path):
    # Get the most recent file in the directory
    files = [f for f in os.listdir(folder_path) if f.endswith('.json') and f.startswith('save')]
    if not files:
        raise FileNotFoundError("No JSON files found in the specified directory")
    
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    file_path = os.path.join(folder_path, latest_file)
    
    # Load the JSON file
    df = pd.read_json(file_path)
    
    # Display basic information about the main DataFrame
    print("\nMain DataFrame Info:")
    print(df.info())
    
    # Display missing values in main DataFrame
    print("\nMissing Values in Main DataFrame:")
    print(df.isnull().sum())
    expanded_tracks_df = None
    # Analyze tracks data
    if 'tracks' in df.columns:
        tracks_data = df['tracks'].iloc[0]  # Get the first row's tracks
        if isinstance(tracks_data, list):
            print(f"\nNumber of tracks: {len(tracks_data)}")
            if len(tracks_data) > 0:
                # First, let's look at the structure of a single track
                print("\nSample Track Structure:")
                print(tracks_data[0])
                
                # Get all unique keys from all tracks
                all_keys = set()
                for track in tracks_data:
                    if isinstance(track, dict):
                        all_keys.update(track.keys())
                
                print("\nAll available track fields:")
                print(sorted(list(all_keys)))
                
                # Create a DataFrame with only the common fields
                common_fields = ['added_at', 'track']  # Add other common fields as needed
                tracks_list = []
                for track in tracks_data:
                    if isinstance(track, dict):
                        track_dict = {field: track.get(field) for field in common_fields}
                        tracks_list.append(track_dict)
                
                tracks_df = pd.DataFrame(tracks_list)
                
                print("\nTracks DataFrame Info:")
                print(tracks_df.info())
                
                print("\nMissing Values in Tracks:")
                print(tracks_df.isnull().sum())
                
                # Analyze the nested track data
                if 'track' in tracks_df.columns:
                    print("\nTrack Object Fields:")
                    first_track = tracks_df['track'].iloc[0]
                    if isinstance(first_track, dict):
                        # Get all track fields
                        track_fields = sorted(list(first_track.keys()))
                        
                        # Create expanded tracks DataFrame directly from tracks_data using a list comprehension
                        expanded_tracks = [
                            {field: track_obj.get(field) for field in track_fields}
                            for track in tracks_data
                            if isinstance(track, dict) and 'track' in track and isinstance(track['track'], dict)
                            for track_obj in [track['track']]
                        ]
                        
                        # Add session and token info to each track dict
                        session = df['session'].iloc[0] if 'session' in df.columns else None
                        token = df['token'].iloc[0] if 'token' in df.columns else None
                        for track_dict in expanded_tracks:
                            track_dict['session'] = session
                            track_dict['token'] = token
                        
                        # Convert to DataFrame
                        expanded_tracks_df = pd.DataFrame(expanded_tracks)
                        print("\nExpanded Tracks DataFrame Info:")
                        print(expanded_tracks_df.info())
    
    if expanded_tracks_df is None:
        raise ValueError("Could not create expanded tracks DataFrame. Check if the data structure matches the expected format.")
        
    return expanded_tracks_df

def preprocess_data(df):
    # For now, we'll skip the preprocessing steps since the data structure is nested
    return df

def main():
    folder_path = "C:\\Users\\hedgi\\Desktop\\spotify_data"
    
    try:
        # Load and analyze data
        df = load_and_analyze_data(folder_path)
        
        # Preprocess data
        processed_df = preprocess_data(df)
        
        # Save processed data as both JSON and parquet for flexibility
        output_json = os.path.join(folder_path, "processed_data.json")
        output_parquet = os.path.join(folder_path, "processed_data.parquet")
        output_csv = os.path.join(folder_path, "processed_data.csv")
        
        processed_df.to_json(output_json)
        processed_df.to_parquet(output_parquet)
        processed_df.to_csv(output_csv)
        print(f"\nProcessed data saved to:")
        print(f"JSON: {output_json}")
        print(f"Parquet: {output_parquet}")
        print(f"CSV: {output_csv}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 