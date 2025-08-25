#!/usr/bin/env python3
"""
Test Earthdata authentication
"""
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_auth():
    token = os.environ.get("EARTHDATA_BEARER")
    if not token:
        print("❌ EARTHDATA_BEARER not found in environment")
        return False
    
    print(f"✅ Token found: {token[:50]}...")
    
    # Test CMR search (should work without auth)
    print("\n🔍 Testing CMR search...")
    try:
        response = requests.get("https://cmr.earthdata.nasa.gov/search/collections.json?short_name=SPL4SMGP")
        if response.status_code == 200:
            print("✅ CMR search successful")
        else:
            print(f"❌ CMR search failed: {response.status_code}")
    except Exception as e:
        print(f"❌ CMR search error: {e}")
    
    # Test authenticated download
    print("\n🔐 Testing authenticated download...")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Try to get a sample file URL first
    try:
        search_response = requests.get(
            "https://cmr.earthdata.nasa.gov/search/granules.json",
            params={
                "short_name": "SPL4SMGP",
                "temporal": "2023-01-01T00:00:00Z,2023-01-01T23:59:59Z",
                "bounding_box": "-102,48,-95,51"
            },
            headers=headers
        )
        
        if search_response.status_code == 200:
            data = search_response.json()
            granules = data.get("feed", {}).get("entry", [])
            if granules:
                print(f"✅ Found {len(granules)} granules")
                
                # Try to get file info
                links = granules[0].get("links", [])
                download_urls = [link["href"] for link in links if link.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#"]
                
                if download_urls:
                    url = download_urls[0]
                    print(f"🔗 Testing download URL: {url[:100]}...")
                    
                    # Test HEAD request
                    head_response = requests.head(url, headers=headers, timeout=10)
                    print(f"📊 HEAD response: {head_response.status_code}")
                    if head_response.status_code in [200, 303]:
                        print("✅ Download URL accessible")
                    else:
                        print(f"❌ Download URL failed: {head_response.status_code}")
                        print(f"Response: {head_response.text[:200]}")
                else:
                    print("❌ No download URLs found")
            else:
                print("❌ No granules found")
        else:
            print(f"❌ Search failed: {search_response.status_code}")
            print(f"Response: {search_response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Authentication test error: {e}")

if __name__ == "__main__":
    test_auth()
