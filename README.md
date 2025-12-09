# Segmentation Dataset Merger (SegMerge)

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**SegMerge** is a robust CLI tool designed to standardize and merge multiple computer vision segmentation datasets into a single, unified format.

Whether you are combining public datasets (COCO) with custom annotations (YOLO .txt) or semantic masks, this tool handles the coordinate normalization, file renaming, and directory structuring automatically.

## 🚀 Features

- **Universal Input Support:** seamlessly reads YOLO (`.txt`), COCO (`.json`), and Semantic Masks (`.png`/`.jpg`).
- **Unified Output:** Converts all inputs into your target format: **YOLO Segmentation** or **Semantic Masks**.
- **Conflict Resolution:** Automatically handles duplicate filenames using UUIDs so data is never overwritten.
- **Coordinate Normalization:** Handles conversion between relative coordinates (0-1) and absolute pixel values.

## 📦 Installation

### Option 1: From Source (Recommended for Developers)

Clone the repository and install it in editable mode:

```bash
git clone [https://github.com/usama-mangi/dataset-merger.git](https://github.com/usama-mangi/dataset-merger.git)
cd dataset-merger
pip install -e .
```
