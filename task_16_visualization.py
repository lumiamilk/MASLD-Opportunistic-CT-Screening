import os
import pandas as pd
import numpy as np
import logging
import SimpleITK as sitk
import openslide
from PIL import Image
import scipy.ndimage as ndimage
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
MASTER_CSV = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
INVENTORY_CSV = os.path.join(BASE_DIR, "output", "task_16_pathology_inventory.csv")
CT_DIR = os.path.join(BASE_DIR, "data", "CT_Cleaned")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "figure_assets_hd_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    try:
        df = pd.read_csv(MASTER_CSV, encoding='utf-8-sig', dtype={'patient_id': str})
    except:
        df = pd.read_csv(MASTER_CSV, encoding='gb18030', dtype={'patient_id': str})
    return df

def get_pathology_path(patient_id):
    if not os.path.exists(INVENTORY_CSV): return None
    inv = pd.read_csv(INVENTORY_CSV, dtype={'Patient_ID': str})
    row = inv[inv['Patient_ID'] == str(patient_id)]
    if not row.empty: return row.iloc[0]['Path']
    return None

def read_dicom_slice(patient_id):
    folder = os.path.join(CT_DIR, str(patient_id))
    if not os.path.exists(folder): return None
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(folder)
    if not series_ids: 
        for root, dirs, files in os.walk(folder):
            series_ids = reader.GetGDCMSeriesIDs(root)
            if series_ids:
                folder = root
                break
        if not series_ids: return None
    dicom_names = reader.GetGDCMSeriesFileNames(folder, series_ids[0])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    arr = sitk.GetArrayFromImage(image)
    
    # Smart Slice Selection
    tissue_counts = []
    for i in range(arr.shape[0]):
        slice_data = arr[i, :, :]
        count = np.sum(slice_data > -200)
        tissue_counts.append(count)
    start = int(arr.shape[0] * 0.2)
    end = int(arr.shape[0] * 0.8)
    if end > start: z = start + np.argmax(tissue_counts[start:end])
    else: z = arr.shape[0] // 2
    return arr[z, :, :]

def window_ct(img, window_center=40, window_width=400):
    min_val = window_center - window_width // 2
    max_val = window_center + window_width // 2
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val) * 255
    return img.astype(np.uint8)

def export_ct_upscaled(img_arr, path, scale=4):
    img_zoom = ndimage.zoom(img_arr, zoom=scale, order=1) 
    im = Image.fromarray(img_zoom)
    im.save(path)

def export_pathology_roi(path_file, output_prefix):
    try:
        slide = openslide.OpenSlide(path_file)
        
        # 1. Find Tissue using HSV Saturation
        # Pick a level that is manageable (width ~2000)
        proc_level = min(3, slide.level_count - 1)
        w, h = slide.level_dimensions[proc_level]
        
        # Read low-res RGB
        thumb_rgb = slide.read_region((0, 0), proc_level, (w, h)).convert("RGB")
        thumb_arr = np.array(thumb_rgb)
        
        # Convert to HSV
        thumb_hsv = rgb2hsv(thumb_arr)
        saturation = thumb_hsv[:, :, 1]
        
        # Otsu Threshold on Saturation (Tissue is colorful, background is gray/white/black)
        try:
            thresh = threshold_otsu(saturation)
        except:
            thresh = 0.05 # Fallback
            
        # Create tissue mask (S > thresh) AND avoid pure black (V > 0.1)
        # Value channel
        value = thumb_hsv[:, :, 2]
        mask = (saturation > thresh) & (value > 0.1)
        
        # Find centroid of tissue
        labeled, n = ndimage.label(mask)
        if n > 0:
            # Find largest component
            sizes = ndimage.sum(mask, labeled, range(n + 1))
            largest = np.argmax(sizes[1:]) + 1
            slices = ndimage.find_objects(labeled)
            roi = slices[largest - 1]
            cy = (roi[0].start + roi[0].stop) // 2
            cx = (roi[1].start + roi[1].stop) // 2
        else:
            # Fallback to center
            cx, cy = w // 2, h // 2
            logging.warning("No tissue detected by HSV, falling back to center.")

        # Save the mask debug image to see what it found
        Image.fromarray((mask * 255).astype(np.uint8)).save(f"{output_prefix}_Mask.jpg")

        # 2. Extract High Res ROI (Level 0)
        # Coordinates from proc_level -> Level 0
        downsample = slide.level_downsamples[proc_level]
        cx_0 = int(cx * downsample)
        cy_0 = int(cy * downsample)
        
        roi_size = 4096
        x_start = max(0, cx_0 - roi_size // 2)
        y_start = max(0, cy_0 - roi_size // 2)
        
        img = slide.read_region((x_start, y_start), 0, (roi_size, roi_size)).convert("RGB")
        img.save(f"{output_prefix}_ROI.jpg", quality=95)
        
        return True
    except Exception as e:
        logging.error(f"Pathology Error: {e}")
        traceback.print_exc()
        return False

def main():
    import traceback
    df = load_data()
    candidates = df[(df['Steatosis_Grade'] >= 3) & (df['Liver_Mean_HU'].notna())].copy()
    candidates = candidates.sort_values('Liver_Mean_HU')
    
    processed = 0
    for _, row in candidates.iterrows():
        if processed >= 3: break
        pid = row['patient_id']
        path_file = get_pathology_path(pid)
        if not path_file or not os.path.exists(path_file): continue
            
        logging.info(f"Processing {pid}")
        
        # CT
        ct = read_dicom_slice(pid)
        if ct is not None:
            ct = window_ct(ct)
            export_ct_upscaled(ct, os.path.join(OUTPUT_DIR, f"CT_{pid}.png"), scale=4)
            
        # Patho
        export_pathology_roi(path_file, os.path.join(OUTPUT_DIR, f"Patho_{pid}"))
        processed += 1

if __name__ == "__main__":
    main()
