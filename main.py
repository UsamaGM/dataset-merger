import os
import json
import argparse
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import uuid

# --- 1. UTILS AND CONVERTERS ---

def normalize_coords(polygon, width, height):
    """Normalizes pixel coordinates to 0-1 range."""
    return [(x / width, y / height) for x, y in polygon]

def denormalize_coords(polygon, width, height):
    """Converts 0-1 range coordinates back to pixels."""
    return [(int(x * width), int(y * height)) for x, y in polygon]

def contour_to_polygon(contour):
    """Flattens OpenCV contour to a list of tuples."""
    return [(float(p[0][0]), float(p[0][1])) for p in contour]

def read_yolo_file(txt_path, img_width, img_height):
    """Reads YOLO segmentation txt and returns list of dicts."""
    annotations = []
    if not os.path.exists(txt_path):
        return annotations
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = list(map(float, line.strip().split()))
        class_id = int(parts[0])
        coords = parts[1:]
        # YOLO format is normalized x y x y...
        polygon = []
        for i in range(0, len(coords), 2):
            polygon.append((coords[i], coords[i+1]))
            
        # Convert to pixels for internal storage
        pixel_poly = denormalize_coords(polygon, img_width, img_height)
        annotations.append({'class_id': class_id, 'segmentation': pixel_poly})
    return annotations

def read_mask_file(mask_path):
    """Reads an image mask and extracts contours as polygons."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    
    annotations = []
    classes = np.unique(mask)
    classes = [c for c in classes if c != 0] # Assume 0 is background
    
    for c in classes:
        # Create binary mask for this class
        binary_mask = np.where(mask == c, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cont in contours:
            if len(cont) < 3: continue # valid polygon needs 3 points
            poly = contour_to_polygon(cont)
            annotations.append({'class_id': int(c), 'segmentation': poly})
            
    return annotations, mask.shape[1], mask.shape[0] # width, height

def read_coco_file(json_path, image_dir):
    """Reads a COCO json and yields image objects."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Map image_id to annotations
    img_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        
        # Handle segmentation (Polygon only supported here, not RLE)
        if isinstance(ann['segmentation'], list):
            for seg in ann['segmentation']:
                # COCO is [x, y, x, y...] flat list
                poly = []
                for i in range(0, len(seg), 2):
                    poly.append((seg[i], seg[i+1]))
                img_to_anns[img_id].append({
                    'class_id': ann['category_id'], # NOTE: Category mapping might be needed
                    'segmentation': poly
                })

    for img in data['images']:
        fname = img['file_name']
        full_path = os.path.join(image_dir, fname)
        anns = img_to_anns.get(img['id'], [])
        yield {
            'path': full_path,
            'width': img['width'],
            'height': img['height'],
            'annotations': anns
        }

# --- 2. WRITERS ---

def write_yolo(output_dir, filename_stem, width, height, annotations):
    """Writes annotations to YOLO format txt."""
    txt_path = os.path.join(output_dir, 'labels', f"{filename_stem}.txt")
    
    with open(txt_path, 'w') as f:
        for ann in annotations:
            cid = ann['class_id']
            # Normalize
            norm_poly = normalize_coords(ann['segmentation'], width, height)
            
            # Clamp values to 0-1 to avoid YOLO errors
            flat_poly = []
            for x, y in norm_poly:
                flat_poly.append(f"{min(max(x, 0), 1):.6f}")
                flat_poly.append(f"{min(max(y, 0), 1):.6f}")
            
            line = f"{cid} {' '.join(flat_poly)}\n"
            f.write(line)

def write_mask(output_dir, filename_stem, width, height, annotations):
    """Writes annotations to a PNG mask."""
    mask_path = os.path.join(output_dir, 'masks', f"{filename_stem}.png")
    
    # Create blank mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for ann in annotations:
        cid = ann['class_id']
        poly = np.array(ann['segmentation'], dtype=np.int32)
        cv2.fillPoly(mask, [poly], int(cid))
        
    cv2.imwrite(mask_path, mask)

# --- 3. MAIN LOGIC ---

def process_datasets(config_path, output_dir, dataset_name, target_format):
    
    with open(config_path, 'r') as f:
        datasets = json.load(f)
        
    final_output_path = os.path.join(output_dir, dataset_name)
    images_out = os.path.join(final_output_path, 'images')
    
    # Setup directories
    if target_format == 'yolo':
        labels_out = os.path.join(final_output_path, 'labels')
        os.makedirs(labels_out, exist_ok=True)
    elif target_format == 'mask':
        masks_out = os.path.join(final_output_path, 'masks')
        os.makedirs(masks_out, exist_ok=True)
    
    os.makedirs(images_out, exist_ok=True)
    
    print(f"--- Merging {len(datasets)} Datasets into '{target_format}' format ---")

    for d_idx, ds in enumerate(datasets):
        base_path = Path(ds['base'])
        img_path = base_path / ds['images']
        lbl_path = base_path / ds['labels']
        
        print(f"Processing Dataset {d_idx+1} at {base_path}...")
        
        # Detect Input Type
        input_type = "unknown"
        if os.path.isdir(lbl_path):
            # check extension of first file
            try:
                first_file = next(os.scandir(lbl_path)).name
                if first_file.endswith('.txt'): input_type = 'yolo'
                elif first_file.endswith(('.png', '.jpg', '.bmp')): input_type = 'mask'
            except StopIteration:
                print(f"Warning: Label directory {lbl_path} is empty.")
                continue
        elif os.path.isfile(lbl_path) and str(lbl_path).endswith('.json'):
            input_type = 'coco'
        
        print(f"Detected format: {input_type}")

        # Iterator for processing files
        file_iterator = []
        
        if input_type == 'yolo' or input_type == 'mask':
            # Collect all images
            valid_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for ext in valid_exts:
                image_files.extend(list(img_path.glob(ext)))
            file_iterator = image_files
            
        elif input_type == 'coco':
            # COCO generator handles its own iteration
            file_iterator = read_coco_file(str(lbl_path), str(img_path))

        # --- PROCESS FILES ---
        for item in tqdm(file_iterator, desc=f"Dataset {d_idx+1}"):
            
            # Extract info based on type
            if input_type in ['yolo', 'mask']:
                src_img_path = str(item)
                stem = item.stem
                
                # Read Image for dims
                img = cv2.imread(src_img_path)
                if img is None: continue
                h, w = img.shape[:2]
                
                # Read Annotations
                anns = []
                if input_type == 'yolo':
                    txt_file = lbl_path / f"{stem}.txt"
                    anns = read_yolo_file(str(txt_file), w, h)
                elif input_type == 'mask':
                    mask_file = lbl_path / f"{stem}.png" # Assuming png for masks
                    if not mask_file.exists(): mask_file = lbl_path / f"{stem}.jpg"
                    if mask_file.exists():
                        anns, _, _ = read_mask_file(str(mask_file))
                        
            elif input_type == 'coco':
                # item is the dict yielded by generator
                src_img_path = item['path']
                w, h = item['width'], item['height']
                anns = item['annotations']
                stem = Path(src_img_path).stem

            # --- WRITE TO OUTPUT ---
            
            # Generate Unique Name to prevent collisions
            unique_stem = f"ds{d_idx}_{stem}_{uuid.uuid4().hex[:4]}"
            ext = Path(src_img_path).suffix
            dest_img_path = os.path.join(images_out, f"{unique_stem}{ext}")
            
            # Copy Image
            shutil.copy2(src_img_path, dest_img_path)
            
            # Write Labels
            if target_format == 'yolo':
                write_yolo(final_output_path, unique_stem, w, h, anns)
            elif target_format == 'mask':
                write_mask(final_output_path, unique_stem, w, h, anns)

    print(f"\nSuccess! Merged dataset saved to: {final_output_path}")

def main():
    parser = argparse.ArgumentParser(description="Merge Segmentation Datasets")
    parser.add_argument('--config', type=str, required=True, help="Path to JSON config file")
    parser.add_argument('--output', type=str, required=True, help="Output directory path")
    parser.add_argument('--name', type=str, default="merged_dataset", help="Name of the new dataset folder")
    parser.add_argument('--format', type=str, required=True, choices=['yolo', 'mask'], help="Target output format")

    args = parser.parse_args()
    process_datasets(args.config, args.output, args.name, args.format)

if __name__ == "__main__":
    main()