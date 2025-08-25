#!/usr/bin/env python3
"""
Sentinel-2 fetcher (stub)
Preferred: earthaccess + CMR/LAADS; or AWS Open Data. Fallback: .netrc/curl.

Usage:
  python3 scripts/fetchers/s2_fetch.py --start 2024-01-01 --end 2024-12-31 --bbox -102,48,-95,51 --cloud 20 --out data/processed/sentinel2

Stub exits with code 2 (not implemented). No mock data.
"""

import argparse
import os
import sys


def check_credentials() -> bool:
    netrc_ok = os.path.exists(os.path.expanduser("~/.netrc"))
    bearer_ok = bool(os.environ.get("EARTHDATA_BEARER"))
    try:
        import earthaccess  # noqa: F401
        ea_ok = True
    except Exception:
        ea_ok = False
    return netrc_ok or bearer_ok or ea_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--bbox", required=True)
    parser.add_argument("--cloud", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not check_credentials():
        print("ERROR: Missing Earthdata credentials")
        return 1

    print("NOT IMPLEMENTED: Real Sentinel-2 download not implemented in stub.")
    print("Args:", vars(args))
    return 2


if __name__ == "__main__":
    sys.exit(main())


