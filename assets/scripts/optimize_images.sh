#!/bin/bash
# This script optimizes images in the specified directory and creates thumbnails.
# Make sure current working directory is assets/scripts

# Input directory
INPUT_DIR="../img"

# Output directories
OPTIMIZED_DIR="../img/optimized"
THUMBNAIL_DIR="../img/thumbnails"

# Create output directories if they don't exist
mkdir -p "$OPTIMIZED_DIR" "$THUMBNAIL_DIR"

# Loop through all .jpg files in the input directory
for img in "$INPUT_DIR"/*.jpg; do
  # Extract the base filename without path or extension
  filename=$(basename "$img")
  base_name="${filename%.jpg}"

  # Convert to WebP (compressed)
  cwebp -q 80 "$img" -o "$OPTIMIZED_DIR/${base_name}.webp"

  # Create thumbnail
  magick "$img" -resize 300x300 "$THUMBNAIL_DIR/${base_name}.webp"

  # Delete original image
  rm "$img"
done