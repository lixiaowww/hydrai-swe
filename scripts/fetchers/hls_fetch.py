#!/usr/bin/env python3
"""
HLS (Harmonized Landsat–Sentinel) fetcher (real)
- Cloud-ready: uses Bearer token directly, no .netrc dependency
- Source: NASA CMR + LAADS DAAC
- Products: HLSL30.002 (Landsat) / HLSS30.002 (Sentinel-2)

Usage:
  python3 scripts/fetchers/hls_fetch.py --start 2024-01-01 --end 2024-12-31 --bbox -102,48,-95,51 --out data/raw/nasa_simple
"""

import argparse
import os
import sys
import requests
import json
from typing import Tuple
from datetime import datetime


def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be: minLon,minLat,maxLon,maxLat")
    return tuple(map(float, parts))  # type: ignore


def search_cmr(short_name: str, start_date: str, end_date: str, bbox: Tuple[float, float, float, float], limit: int = 50):
    """Search CMR for HLS granules"""
    token = os.environ.get("EARTHDATA_BEARER")
    if not token:
        raise Exception("EARTHDATA_BEARER environment variable required")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # CMR search parameters
    params = {
        "short_name": short_name,
        "temporal": f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
        "bounding_box": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    }
    
    url = "https://cmr.earthdata.nasa.gov/search/granules.json"
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"CMR search failed: {response.status_code} - {response.text}")
    
    data = response.json()
    granules = data.get("feed", {}).get("entry", [])
    
    if not granules:
        print(f"WARN: No {short_name} granules found for {start_date} to {end_date}")
        return []
    
    print(f"Found {len(granules)} granules")
    return granules


def download_granules(granules: list, out_dir: str):
    """Download HLS granules to output directory"""
    token = os.environ.get("EARTHDATA_BEARER")
    headers = {"Authorization": f"Bearer {token}"}
    
    downloaded_files = []
    
    for granule in granules:
        try:
            # Get download URLs from granule
            links = granule.get("links", [])
            download_urls = [link["href"] for link in links if link.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#"]
            
            if not download_urls:
                continue
                
            # Download first available file
            url = download_urls[0]
            filename = os.path.basename(url.split("?")[0])  # Remove query params
            filepath = os.path.join(out_dir, filename)
            
            print(f"Downloading {filename}...")
            response = requests.get(url, headers=headers, stream=True)
            
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded_files.append(filepath)
                print(f"✓ Downloaded {filename}")
            else:
                print(f"✗ Failed to download {filename}: {response.status_code}")
                
        except Exception as e:
            print(f"✗ Error downloading granule: {e}")
            continue
    
    return downloaded_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--bbox", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--product", default="HLSL30.002")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    try:
        bbox = parse_bbox(args.bbox)
    except Exception as e:
        print(f"ERROR: invalid bbox: {e}")
        return 1

    os.makedirs(args.out, exist_ok=True)

    try:
        # Search CMR
        granules = search_cmr(args.product, args.start, args.end, bbox, args.limit)
        if not granules:
            return 3
        
        # Download granules
        downloaded = download_granules(granules, args.out)
        print(f"Successfully downloaded {len(downloaded)} files to {args.out}")
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())


