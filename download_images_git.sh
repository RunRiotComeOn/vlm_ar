#!/bin/bash
# Download Vision-SR1-Cold-9K images using git clone
# This method avoids HuggingFace API rate limits

echo "============================================================"
echo "Downloading Vision-SR1-Cold-9K using Git LFS"
echo "============================================================"

# Set up directories
WORK_DIR="/nas03/yixuh/vlm-adaptive-resoning"
TEMP_DIR="$WORK_DIR/.temp_cold_start_download"
TARGET_DIR="$WORK_DIR/LLaMA-Factory/data/cold_start_9k/images"

# Clean up old temp directory if exists
if [ -d "$TEMP_DIR" ]; then
    echo "Removing old temporary directory..."
    rm -rf "$TEMP_DIR"
fi

# Create target directory
mkdir -p "$TARGET_DIR"

echo ""
echo "Step 1: Cloning repository..."
echo "This may take some time as it downloads ~9K images"
echo ""

cd "$WORK_DIR"

# Clone the dataset repository
git clone https://huggingface.co/datasets/LMMs-Lab-Turtle/Vision-SR1-Cold-9K "$TEMP_DIR"

if [ $? -ne 0 ]; then
    echo "✗ Failed to clone repository"
    exit 1
fi

echo ""
echo "Step 2: Locating images..."
echo ""

# Find where images are stored
if [ -d "$TEMP_DIR/cold_start" ]; then
    IMAGE_SOURCE="$TEMP_DIR/cold_start"
    echo "Found images in: cold_start/"
elif [ -d "$TEMP_DIR/images" ]; then
    IMAGE_SOURCE="$TEMP_DIR/images"
    echo "Found images in: images/"
elif [ -d "$TEMP_DIR/data/images" ]; then
    IMAGE_SOURCE="$TEMP_DIR/data/images"
    echo "Found images in: data/images/"
else
    echo "✗ Could not find images directory"
    echo "Directory structure:"
    ls -la "$TEMP_DIR"
    exit 1
fi

echo ""
echo "Step 3: Copying images to target directory..."
echo ""

# Copy images
cp "$IMAGE_SOURCE"/*.jpg "$TARGET_DIR/" 2>/dev/null
cp "$IMAGE_SOURCE"/*.png "$TARGET_DIR/" 2>/dev/null

# Count copied images
IMAGE_COUNT=$(ls -1 "$TARGET_DIR" | wc -l)

echo ""
echo "Step 4: Cleanup..."
echo ""

# Remove temporary directory
rm -rf "$TEMP_DIR"

echo ""
echo "============================================================"
echo "Download Complete"
echo "============================================================"
echo "Copied $IMAGE_COUNT images to: $TARGET_DIR"
echo ""
echo "Next steps:"
echo "  python verify_cold_start_data.py"
echo ""
