#!/usr/bin/env python3
"""
Sentinel-1 SAR fetcher (stub)
Preferred: Copernicus SciHub / ASF Vertex / AWS Open Data

Usage:
  python3 scripts/fetchers/s1_fetch.py --start 2024-01-01 --end 2024-12-31 --bbox -102,48,-95,51 --out data/raw/sentinel1

Stub exits with code 2 (not implemented).
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--bbox", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print("NOT IMPLEMENTED: Real Sentinel-1 download not implemented in stub.")
    print("Args:", vars(args))
    return 2


if __name__ == "__main__":
    sys.exit(main())


