"""
fetch_data.py
Fetches video statistics from YouTube Data API v3
for one or multiple channels and saves to CSV.
"""

import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")


def get_youtube_client():
    return build("youtube", "v3", developerKey=API_KEY)


def get_channel_id_from_handle(youtube, handle):
    """
    Converts a channel handle like @JPFans or a channel name
    to a channel ID. Also accepts a raw channel ID directly.
    """
    # If it already looks like a channel ID, return it
    if handle.startswith("UC") and len(handle) == 24:
        return handle

    # Search for the channel
    request = youtube.search().list(
        part="snippet",
        q=handle,
        type="channel",
        maxResults=1
    )
    response = request.execute()

    if response["items"]:
        channel_id = response["items"][0]["snippet"]["channelId"]
        channel_title = response["items"][0]["snippet"]["title"]
        print(f"Found channel: {channel_title} ({channel_id})")
        return channel_id
    else:
        raise ValueError(f"Could not find channel: {handle}")


def get_uploads_playlist_id(youtube, channel_id):
    """Gets the uploads playlist ID for a channel."""
    request = youtube.channels().list(
        part="contentDetails,snippet,statistics",
        id=channel_id
    )
    response = request.execute()
    channel_info = response["items"][0]

    playlist_id = channel_info["contentDetails"]["relatedPlaylists"]["uploads"]
    channel_name = channel_info["snippet"]["title"]
    subscriber_count = channel_info["statistics"].get("subscriberCount", 0)

    return playlist_id, channel_name, int(subscriber_count)


def get_all_video_ids(youtube, playlist_id, max_videos=200):
    """
    Fetches all video IDs from an uploads playlist.
    max_videos: cap to avoid burning API quota (default 200)
    """
    video_ids = []
    next_page_token = None

    while len(video_ids) < max_videos:
        request = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response["items"]:
            video_ids.append(item["contentDetails"]["videoId"])

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.1)  # gentle rate limiting

    return video_ids[:max_videos]


def get_video_details(youtube, video_ids, channel_name):
    """
    Fetches detailed stats + metadata for a list of video IDs.
    YouTube API allows up to 50 IDs per request.
    """
    all_videos = []

    # Process in batches of 50
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(batch)
        )
        response = request.execute()

        for item in response["items"]:
            snippet = item["snippet"]
            stats = item.get("statistics", {})
            content = item["contentDetails"]

            # Parse duration from ISO 8601 (PT4M13S → seconds)
            duration_str = content.get("duration", "PT0S")
            duration_seconds = parse_duration(duration_str)

            video = {
                "channel_name": channel_name,
                "video_id": item["id"],
                "title": snippet.get("title", ""),
                "description": snippet.get("description", "")[:500],  # truncate
                "published_at": snippet.get("publishedAt", ""),
                "tags": "|".join(snippet.get("tags", [])),
                "category_id": snippet.get("categoryId", ""),
                "duration_seconds": duration_seconds,
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "comment_count": int(stats.get("commentCount", 0)),
                "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                "has_custom_thumbnail": "maxres" in str(snippet.get("thumbnails", {})),
            }
            all_videos.append(video)

        time.sleep(0.1)

    return all_videos


def parse_duration(duration_str):
    """Converts ISO 8601 duration (PT4M13S) to total seconds."""
    import re
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def fetch_channel(handle, max_videos=200):
    """
    Main function to fetch all video data for one channel.
    Returns a DataFrame.
    """
    youtube = get_youtube_client()

    print(f"\nFetching data for: {handle}")
    channel_id = get_channel_id_from_handle(youtube, handle)
    playlist_id, channel_name, subscribers = get_uploads_playlist_id(youtube, channel_id)

    print(f"Channel: {channel_name} | Subscribers: {subscribers:,}")
    print(f"Fetching up to {max_videos} videos...")

    video_ids = get_all_video_ids(youtube, playlist_id, max_videos)
    print(f"Found {len(video_ids)} videos. Fetching details...")

    videos = get_video_details(youtube, video_ids, channel_name)
    df = pd.DataFrame(videos)
    df["subscriber_count"] = subscribers

    print(f"Done. {len(df)} videos fetched.")
    return df


def fetch_multiple_channels(channel_handles, max_videos_each=200, save_path="data/raw/videos.csv"):
    """
    Fetches data for multiple channels and combines into one DataFrame.
    channel_handles: list of channel names, handles, or IDs
    """
    all_dfs = []

    for handle in channel_handles:
        try:
            df = fetch_channel(handle, max_videos_each)
            all_dfs.append(df)
            time.sleep(1)  # pause between channels
        except Exception as e:
            print(f"Error fetching {handle}: {e}")
            continue

    if not all_dfs:
        raise ValueError("No data fetched. Check your API key and channel names.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(save_path, index=False)
    print(f"\nSaved {len(combined)} total videos to {save_path}")
    return combined


# ── Run this directly to fetch data ──────────────────────────────────────────
if __name__ == "__main__":
    # Replace these with your channel and competitor channels
    CHANNELS = [
        "Pewdiepie",       # your channel — replace this
        "Gigguk",                    # competitor example
        "TrashTasteShow",            # competitor example
    ]

    df = fetch_multiple_channels(
        channel_handles=CHANNELS,
        max_videos_each=200,
        save_path="data/raw/videos.csv"
    )

    print("\nSample data:")
    print(df[["channel_name", "title", "view_count", "like_count"]].head(10))