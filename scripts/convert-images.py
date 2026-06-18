#!/usr/bin/env python3
"""Convert all PNG images in assets/ to WebP format.

Usage:
    python3 scripts/convert-images.py

Requires Pillow: pip3 install Pillow
"""

import os
import sys

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install with: pip3 install Pillow")
    sys.exit(1)

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
QUALITY = 80

def convert_png_to_webp():
    converted = 0
    skipped = 0
    errors = 0

    for dirpath, dirnames, filenames in os.walk(ASSETS_DIR):
        for fname in filenames:
            if not fname.lower().endswith(".png"):
                continue

            png_path = os.path.join(dirpath, fname)
            webp_path = png_path.rsplit(".", 1)[0] + ".webp"

            if os.path.exists(webp_path):
                skipped += 1
                continue

            try:
                img = Image.open(png_path)
                img.save(webp_path, "WEBP", quality=QUALITY, method=6)
                converted += 1
                png_size = os.path.getsize(png_path)
                webp_size = os.path.getsize(webp_path)
                ratio = (1 - webp_size / png_size) * 100
                print(f"  {os.path.relpath(png_path)}: {png_size//1024}KB -> {webp_size//1024}KB ({ratio:.0f}% smaller)")
            except Exception as e:
                print(f"  Error: {png_path}: {e}")
                errors += 1

    print(f"\nDone: {converted} converted, {skipped} already exist, {errors} errors")

if __name__ == "__main__":
    convert_png_to_webp()
