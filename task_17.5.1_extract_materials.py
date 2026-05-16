import os
import glob
import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
import openslide
from PIL import Image
import scipy.ndimage as ndimage
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

# --- Configuration ---
BASE_DIR = os.getcwd()
if "/mnt/d" not in BASE_DIR and "paper0" not in BASE_DIR:
     BASE_DIR = "/mnt/d/mWork/paper0"

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CT_DIR = os.path.join(BASE_DIR, "data", "CT_Cleaned", "2120033358")
HE_PATH = os.path.join(BASE_DIR, "data", "临床liver 切片", "临床 liver HE", "III级脂肪肝 L210129.mrxs")
RET_PATH = os.path.join(BASE_DIR, "data", "临床liver 切片", "临床 liver 网染", "III级脂肪肝 L210129.mrxs")

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_ct_slice():
    logging.info("Processing CT...")
    if not os.path.exists(CT_DIR):
        logging.error(f"CT Directory not found: {CT_DIR}")
        return

    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(CT_DIR)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        arr = sitk.GetArrayFromImage(image) # (Z, Y, X)
        logging.info(f"CT Shape: {arr.shape}")
        
        # Find best slice (Max tissue area)
        # Tissue range approx -150 to 200 (exclude air and dense bone/contrast if any)
        # But mostly we want to avoid lungs (<-500).
        # Simple count of pixels > -200
        tissue_counts = []
        for z in range(arr.shape[0]):
            slice_arr = arr[z, :, :]
            count = np.sum(slice_arr > -200)
            tissue_counts.append(count)
        
        # Heuristic: Focus on middle 60% to avoid pelvis/neck if whole body
        start_z = int(arr.shape[0] * 0.2)
        end_z = int(arr.shape[0] * 0.8)
        
        best_z = start_z + np.argmax(tissue_counts[start_z:end_z])
        logging.info(f"Selected Slice Z={best_z}")
        
        best_slice = arr[best_z, :, :]
        
        # Windowing (Abdomen: Center 40, Width 400 -> Min -160, Max 240)
        wc, ww = 40, 400
        min_val = wc - ww / 2
        max_val = wc + ww / 2
        
        img_windowed = np.clip(best_slice, min_val, max_val)
        img_norm = (img_windowed - min_val) / (max_val - min_val)
        img_uint8 = (img_norm * 255).astype(np.uint8)
        
        # Save
        save_path = os.path.join(OUTPUT_DIR, "task17.5.1_CT_Original.png")
        Image.fromarray(img_uint8).save(save_path)
        logging.info(f"Saved CT to {save_path}")
        
    except Exception as e:
        logging.error(f"CT Extraction Failed: {e}")

def find_multiple_rois(slide, n=5, roi_size_l0=2048):
    """
    Finds n non-overlapping ROIs with high tissue density.
    """
    try:
        # 1. Select downsample level
        best_level = 0
        for i, (w, h) in enumerate(slide.level_dimensions):
            if 1000 <= w <= 5000:
                best_level = i
                break
            if w < 1000:
                best_level = max(0, i - 1)
                break
            best_level = i # Fallback

        search_w, search_h = slide.level_dimensions[best_level]
        downsample = slide.level_downsamples[best_level]
        logging.info(f"ROI Search Level {best_level}: {search_w}x{search_h}, Downsample: {downsample}")

        # 2. Get Overview & Mask
        img = slide.read_region((0, 0), best_level, (search_w, search_h)).convert("RGB")
        img_arr = np.array(img)
        
        hsv = rgb2hsv(img_arr)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        
        # Tissue Mask
        try:
            thresh = threshold_otsu(sat)
            thresh = max(0.05, min(thresh, 0.2))
        except:
            thresh = 0.05
            
        tissue_mask = (sat > thresh) & (val > 0.1)
        tissue_mask = ndimage.binary_fill_holes(tissue_mask)
        
        # 3. Density Map
        kernel_size = int(roi_size_l0 / downsample)
        if kernel_size < 1: kernel_size = 1
        
        # Use uniform filter as a "sliding window sum" proxy
        density_map = ndimage.uniform_filter(tissue_mask.astype(float), size=kernel_size, mode='constant')
        
        # 4. Iteratively find peaks
        found_rois = []
        
        # Mask out edges to avoid incomplete ROIs
        edge_margin = kernel_size // 2
        density_map[:edge_margin, :] = 0
        density_map[-edge_margin:, :] = 0
        density_map[:, :edge_margin] = 0
        density_map[:, -edge_margin:] = 0
        
        for i in range(n):
            if np.max(density_map) < 0.1: # Threshold for "good enough" tissue
                break
                
            flat_idx = np.argmax(density_map)
            cy_s, cx_s = np.unravel_index(flat_idx, density_map.shape)
            
            # Convert to Level 0
            center_x_l0 = int(cx_s * downsample)
            center_y_l0 = int(cy_s * downsample)
            
            top_left_x = max(0, center_x_l0 - roi_size_l0 // 2)
            top_left_y = max(0, center_y_l0 - roi_size_l0 // 2)
            
            # Check bounds
            if top_left_x + roi_size_l0 > slide.dimensions[0]: top_left_x = slide.dimensions[0] - roi_size_l0
            if top_left_y + roi_size_l0 > slide.dimensions[1]: top_left_y = slide.dimensions[1] - roi_size_l0
            
            found_rois.append((top_left_x, top_left_y))
            
            # Zero out this region in density map to force next ROI to be elsewhere
            # We zero out a region slightly larger than ROI size to ensure separation
            exclusion_radius = int(kernel_size * 1.2)
            y_min = max(0, cy_s - exclusion_radius)
            y_max = min(density_map.shape[0], cy_s + exclusion_radius)
            x_min = max(0, cx_s - exclusion_radius)
            x_max = min(density_map.shape[1], cx_s + exclusion_radius)
            
            density_map[y_min:y_max, x_min:x_max] = 0
            
        return found_rois
        
    except Exception as e:
        logging.error(f"Error in ROI Search: {e}")
        return []

def process_he_rois():
    logging.info("Processing HE Slide...")
    if not os.path.exists(HE_PATH):
        logging.error(f"HE File not found: {HE_PATH}")
        return

    slide = openslide.OpenSlide(HE_PATH)
    try:
        rois = find_multiple_rois(slide, n=5, roi_size_l0=2048)
        logging.info(f"Found {len(rois)} HE ROIs.")
        
        for idx, (x, y) in enumerate(rois):
            # Read ROI
            roi_img = slide.read_region((x, y), 0, (2048, 2048)).convert("RGB")
            roi_arr = np.array(roi_img)
            
            # --- V5.3 Fat Segmentation Logic ---
            hsv = rgb2hsv(roi_arr)
            S = hsv[:, :, 1]
            V = hsv[:, :, 2]
            
            stroma_mask = S > 0.05
            whole_tissue_mask = ndimage.binary_fill_holes(stroma_mask)
            whole_tissue_mask = ndimage.binary_closing(whole_tissue_mask, structure=np.ones((5,5)))
            
            fat_candidates = (S < 0.1) & (V > 0.85)
            real_fat_mask = fat_candidates & whole_tissue_mask
            
            label_img = label(real_fat_mask)
            refined_fat_mask = np.zeros_like(real_fat_mask, dtype=bool)
            
            for prop in regionprops(label_img):
                is_standard = (50 < prop.area < 5000) and (prop.eccentricity < 0.92) and (prop.solidity > 0.85)
                is_giant = (5000 <= prop.area < 50000) and (prop.minor_axis_length > 50) and (prop.eccentricity < 0.90) and (prop.solidity > 0.80)
                
                if is_standard or is_giant:
                    refined_fat_mask[prop.coords[:, 0], prop.coords[:, 1]] = True
            
            # --- Save Original ---
            save_path_orig = os.path.join(OUTPUT_DIR, f"task17.5.1_HE_ROI_{idx}.jpg")
            roi_img.save(save_path_orig, quality=95)
            
            # --- Save Mask Overlay ---
            vis_img = roi_arr.copy()
            # Green Overlay for Fat
            vis_img[refined_fat_mask] = [0, 255, 0]
            
            save_path_mask = os.path.join(OUTPUT_DIR, f"task17.5.1_HE_Mask_{idx}.jpg")
            Image.fromarray(vis_img).save(save_path_mask, quality=95)
            
            logging.info(f"Saved HE ROI {idx}")
            
    finally:
        slide.close()

def process_reticulin_rois():
    logging.info("Processing Reticulin Slide...")
    if not os.path.exists(RET_PATH):
        logging.error(f"Reticulin File not found: {RET_PATH}")
        return

    slide = openslide.OpenSlide(RET_PATH)
    try:
        rois = find_multiple_rois(slide, n=3, roi_size_l0=2048)
        logging.info(f"Found {len(rois)} Reticulin ROIs.")
        
        for idx, (x, y) in enumerate(rois):
            # Read ROI
            roi_img = slide.read_region((x, y), 0, (2048, 2048)).convert("RGB")
            
            # Save Original
            save_path = os.path.join(OUTPUT_DIR, f"task17.5.1_Ret_ROI_{idx}.jpg")
            roi_img.save(save_path, quality=95)
            
            logging.info(f"Saved Ret ROI {idx}")
            
    finally:
        slide.close()

if __name__ == "__main__":
    extract_ct_slice()
    process_he_rois()
    process_reticulin_rois()
