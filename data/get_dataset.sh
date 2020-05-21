#!/bin/bash

URL = https://drive.google.com/uc?export=download&confirm=pO_n&id=1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF
ZIP_FILE=selfie2anime.zip
TARGET_DIR=selfie2anime
wget ${URL}
unzip ${ZIP_FILE}
rm ${ZIP_FILE}

# Adapt to project expected directory heriarchy
mkdir "$TARGET_DIR"
mkdir -p "$TARGET_DIR/train" "$TARGET_DIR/test"
mv "trainA" "$TARGET_DIR/train/A"
mv "trainB" "$TARGET_DIR/train/B"
mv "testA" "$TARGET_DIR/test/A"
mv "testB" "$TARGET_DIR/test/B"
