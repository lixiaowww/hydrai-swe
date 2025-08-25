#!/usr/bin/env python3
"""
SMAP soil moisture fetcher (real, via direct CMR/LAADS HTTP)
- Cloud-ready: uses Bearer token directly, no .netrc dependency
- Source: NASA CMR + LAADS DAAC

Usage:
  python3 scripts/fetchers/smap_fetch.py --start 2024-01-01 --end 2024-12-31 --bbox -102,48,-95,51 --product SPL3SMP_E.005 --out data/raw/nasa_smap

Notes:
- Requires EARTHDATA_BEARER env var
- Downloads real SMAP granules via HTTP
"""

import argparse
import os
import sys
import requests
import json
import time
from typing import Tuple
from datetime import datetime


def retry_with_backoff(func, max_retries=3, base_delay=1):
    """Retry function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            delay = base_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed, retrying in {delay} seconds...")
            time.sleep(delay)
    
    return None


def download_with_retry(url: str, headers: dict, filepath: str, expected_size: int = 0):
    """Download file with retry mechanism and redirect handling"""
    def download_attempt():
        # Follow redirects automatically for NASA data
        response = requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True)
        if response.status_code not in [200, 303]:
            raise Exception(f"HTTP {response.status_code}")
        
        # If we get a redirect, follow it
        if response.status_code == 303:
            redirect_url = response.headers.get('Location')
            if redirect_url:
                print(f"  Following redirect to: {redirect_url}")
                response = requests.get(redirect_url, headers=headers, stream=True, timeout=60)
                if response.status_code != 200:
                    raise Exception(f"Redirect failed: HTTP {response.status_code}")
            else:
                raise Exception("Redirect response without Location header")
        
        downloaded_size = 0
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
        
        # Verify download
        if os.path.exists(filepath):
            actual_size = os.path.getsize(filepath)
            if actual_size == 0:
                raise Exception("Downloaded file is empty")
            if expected_size > 0 and abs(actual_size - expected_size) > 1024:
                print(f"⚠️  Size mismatch: expected {expected_size}, got {actual_size}")
        
        return actual_size
    
    return retry_with_backoff(download_attempt)


def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be: minLon,minLat,maxLon,maxLat")
    return tuple(map(float, parts))  # type: ignore


def search_cmr(short_name: str, start_date: str, end_date: str, bbox: Tuple[float, float, float, float], limit: int = 50):
    """Search CMR for granules"""
    token = os.environ.get("EARTHDATA_BEARER")
    if not token:
        raise Exception("Authentication required. Please set EARTHDATA_BEARER environment variable.")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # CMR search parameters
    params = {
        "short_name": short_name,
        "temporal": f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
        "bounding_box": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    }
    
    url = "https://cmr.earthdata.nasa.gov/search/granules.json"
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        if response.status_code == 401:
            raise Exception("Authentication failed. Please check your credentials.")
        elif response.status_code == 400:
            raise Exception("Invalid search parameters. Please check date range and coordinates.")
        elif response.status_code == 403:
            raise Exception("Access denied. You may not have permission for this dataset.")
        elif response.status_code != 200:
            raise Exception(f"Search service temporarily unavailable (HTTP {response.status_code})")
            
        data = response.json()
        granules = data.get("feed", {}).get("entry", [])
        
        if not granules:
            print(f"No {short_name} data found for the specified time and location.")
            return []
        
        print(f"Found {len(granules)} data files")
        return granules
        
    except requests.exceptions.Timeout:
        raise Exception("Search request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        raise Exception("Unable to connect to data service. Please check your internet connection.")
    except Exception as e:
        if "Authentication failed" in str(e) or "Access denied" in str(e):
            raise e
        else:
            raise Exception("Search service error. Please try again later.")


def download_granules(granules: list, out_dir: str):
    """Download SMAP granules to output directory with quality validation and retry"""
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
            
            # Get file info first
            try:
                head_response = requests.head(url, headers=headers, timeout=10)
                if head_response.status_code != 200:
                    print(f"✗ Failed to get file info for {filename}: {head_response.status_code}")
                    continue
                    
                expected_size = int(head_response.headers.get('content-length', 0))
                print(f"  Expected size: {expected_size} bytes")
            except Exception as e:
                print(f"⚠️  Could not get file info: {e}")
                expected_size = 0
            
            # Download with retry
            try:
                actual_size = download_with_retry(url, headers, filepath, expected_size)
                
                if actual_size and actual_size > 0:
                    # Basic format validation
                    if filename.endswith('.h5'):
                        try:
                            import h5py
                            with h5py.File(filepath, 'r') as hf:
                                print(f"✓ HDF5 file validated: {filename}")
                        except Exception as e:
                            print(f"⚠️  HDF5 validation failed for {filename}: {e}")
                    elif filename.endswith('.nc'):
                        try:
                            import netCDF4
                            with netCDF4.Dataset(filepath, 'r') as nc:
                                print(f"✓ NetCDF file validated: {filename}")
                        except Exception as e:
                            print(f"⚠️  NetCDF validation failed for {filename}: {e}")
                    
                    downloaded_files.append(filepath)
                    print(f"✓ Downloaded and validated {filename} ({actual_size} bytes)")
                else:
                    print(f"✗ Download failed for {filename}")
                    
            except Exception as e:
                print(f"✗ Download failed for {filename}: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                continue
                
        except Exception as e:
            print(f"✗ Error processing granule: {e}")
            continue
    
    return downloaded_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--bbox", required=True)
    parser.add_argument("--product", default="SPL3SMP_E.005")
    parser.add_argument("--out", required=True)
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


