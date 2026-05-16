import os
import pandas as pd
import numpy as np
import logging
import openslide
import scipy.ndimage as ndimage
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import concurrent.futures
from tqdm import tqdm
import time

# --- Configuration ---
BASE_DIR = os.getcwd()
if "/mnt/d" not in BASE_DIR and "paper0" not in BASE_DIR:
     BASE_DIR = "/mnt/d/mWork/paper0"

MASTER_CSV = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
INVENTORY_CSV = os.path.join(BASE_DIR, "output", "task_16_pathology_inventory.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "figure_assets_hd_v5.3")
OUTPUT_CSV = os.path.join(BASE_DIR, "output", "task_16_quant_results_v5.3.csv")
REPORT_MD = os.path.join(BASE_DIR, "output", "task_16_quant_report_v5.3.md")
LOG_FILE = os.path.join(BASE_DIR, "output", "task_16_quant_v5.3.log")

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)

def get_inventory():
    if not os.path.exists(INVENTORY_CSV):
        logging.error(f"Inventory file not found: {INVENTORY_CSV}")
        return None
    return pd.read_csv(INVENTORY_CSV, dtype={'Patient_ID': str})

def find_best_roi(slide, target_roi_size_l0=4096):
    """
    Finds the best ROI (highest tissue density) using a downsampled layer.
    """
    try:
        best_level = 0
        for i, (w, h) in enumerate(slide.level_dimensions):
            if 1000 <= w <= 5000:
                best_level = i
                break
            if w < 1000:
                best_level = max(0, i - 1)
                break
            best_level = i

        search_w, search_h = slide.level_dimensions[best_level]
        downsample = slide.level_downsamples[best_level]
        
        if search_w * search_h > 50_000_000: 
            return None, None

        img = slide.read_region((0, 0), best_level, (search_w, search_h)).convert("RGB")
        img_arr = np.array(img)
        
        hsv = rgb2hsv(img_arr)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        
        try: 
            thresh = threshold_otsu(sat)
            thresh = max(0.05, min(thresh, 0.2))
        except:
            thresh = 0.05
            
        tissue_mask = (sat > thresh) & (val > 0.1)
        tissue_mask = ndimage.binary_fill_holes(tissue_mask)
        
        if np.sum(tissue_mask) < 1000:
            return None, None

        kernel_size = int(target_roi_size_l0 / downsample)
        if kernel_size < 1: kernel_size = 1
        density_map = ndimage.uniform_filter(tissue_mask.astype(float), size=kernel_size, mode='constant')
        
        flat_idx = np.argmax(density_map)
        cy_s, cx_s = np.unravel_index(flat_idx, density_map.shape)
        
        center_x_l0 = int(cx_s * downsample)
        center_y_l0 = int(cy_s * downsample)
        
        top_left_x = max(0, center_x_l0 - target_roi_size_l0 // 2)
        top_left_y = max(0, center_y_l0 - target_roi_size_l0 // 2)
        
        return top_left_x, top_left_y

    except Exception as e:
        logging.error(f"Error in ROI Search: {e}")
        return None, None

def process_patient_worker(args):
    """
    Worker function with Task 16.3 V4: Precision Fat Segmentation.
    """
    patient_id, path_file = args
    roi_size = 4096
    slide = None
    
    try:
        if not os.path.exists(path_file):
            return {'Patient_ID': patient_id, 'Status': 'File_Not_Found'}

        slide = openslide.OpenSlide(path_file)
        x, y = find_best_roi(slide, target_roi_size_l0=roi_size)
        
        if x is None:
            return {'Patient_ID': patient_id, 'Status': 'No_Tissue_Found'}
            
        w_slide, h_slide = slide.dimensions
        if x + roi_size > w_slide: x = w_slide - roi_size
        if y + roi_size > h_slide: y = h_slide - roi_size
        x, y = max(0, x), max(0, y)
        
        roi_img = slide.read_region((x, y), 0, (roi_size, roi_size)).convert("RGB")
        roi_arr = np.array(roi_img)
        
        hsv = rgb2hsv(roi_arr)
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]
        
        # 1. Whole Tissue Mask
        stroma_mask = S > 0.05
        whole_tissue_mask = ndimage.binary_fill_holes(stroma_mask)
        # binary closing to merge tiny islands
        whole_tissue_mask = ndimage.binary_closing(whole_tissue_mask, structure=np.ones((5,5)))
        
        # 2. Initial Fat Candidates
        fat_candidates = (S < 0.1) & (V > 0.85)
        real_fat_mask = fat_candidates & whole_tissue_mask
        
        # 3. Precision Filtering (V5 Hierarchical Logic)
        label_img = label(real_fat_mask)
        refined_fat_mask = np.zeros_like(real_fat_mask, dtype=bool)
        
        for prop in regionprops(label_img):
            # Condition 1: Standard Vacuoles
            is_standard = (50 < prop.area < 5000) and \
                          (prop.eccentricity < 0.92) and \
                          (prop.solidity > 0.85)
            
            # Condition 2: Giant / Macrovesicular Vacuoles (V5.1 Stricter)
            # Added upper bound 50000 to exclude huge veins
            # Tightened solidity to 0.80 to exclude irregular shapes
            # Added eccentricity check to exclude long giant vessels
            is_giant = (5000 <= prop.area < 50000) and \
                       (prop.minor_axis_length > 50) and \
                       (prop.eccentricity < 0.90) and \
                       (prop.solidity > 0.80)
            
            if is_standard or is_giant:
                refined_fat_mask[prop.coords[:, 0], prop.coords[:, 1]] = True
            
        total_tissue_pixels = np.sum(whole_tissue_mask)
        fat_pixels = np.sum(refined_fat_mask)
        
        fat_ratio = 0.0
        if total_tissue_pixels > 0:
            fat_ratio = fat_pixels / total_tissue_pixels
            
        # --- Visualization ---
        vis_img = roi_arr.copy()
        # Mark refined fat in bright Green
        vis_img[refined_fat_mask] = [0, 255, 0]
        
        save_path = os.path.join(OUTPUT_DIR, f"{patient_id}_ROI_V4.jpg")
        Image.fromarray(vis_img).resize((1024, 1024)).save(save_path, quality=85)
        
        return {
            'Patient_ID': patient_id,
            'Status': 'Success',
            'Fat_Ratio': fat_ratio,
            'Tissue_Pixels': total_tissue_pixels,
            'Fat_Pixels': fat_pixels,
            'ROI_X': x,
            'ROI_Y': y
        }

    except Exception as e:
        logging.error(f"Failed processing {patient_id}: {e}")
        return {'Patient_ID': patient_id, 'Status': 'Error', 'Error_Msg': str(e)}
    finally:
        if slide: slide.close()

def run_analysis(df_res):
    """
    Merges results with master dataset and calculates correlations.
    """
    if df_res.empty: return
    
    df_master = pd.read_csv(MASTER_CSV, dtype={'patient_id': str})
    df_master['patient_id'] = df_master['patient_id'].str.strip()
    
    # Merge
    merged = pd.merge(df_res, df_master, left_on='Patient_ID', right_on='patient_id', how='inner')
    
    if merged.empty:
        logging.warning("No overlap between quantified results and master dataset.")
        return

    # Metrics to correlate
    targets = ['Liver_Mean_HU', 'Steatosis_Grade', 'NAS', 'SAF']
    results_stats = []
    
    plt.figure(figsize=(15, 10))
    for i, target in enumerate(targets):
        if target not in merged.columns: continue
        
        valid = merged.dropna(subset=['Fat_Ratio', target])
        if len(valid) < 3: continue
        
        r, p = stats.pearsonr(valid['Fat_Ratio'], valid[target])
        results_stats.append({'Metric': target, 'Pearson_R': r, 'P_Value': p, 'N': len(valid)})
        
        plt.subplot(2, 2, i+1)
        sns.regplot(data=valid, x='Fat_Ratio', y=target, scatter_kws={'alpha':0.5})
        plt.title(f"Fat_Ratio vs {target}\nN={len(valid)}, R={r:.3f}, p={p:.3e}")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "output", "task_16_v4_correlations.png"))
    
    # Generate Markdown Report
    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write("# Task 16.3 V4: Pathology Quantification Final Report\n\n")
        f.write("## 1. Correlation Analysis\n")
        f.write(pd.DataFrame(results_stats).to_markdown(index=False))
        f.write("\n\n## 2. Top 15 Samples\n")
        cols = ['Patient_ID', 'Fat_Ratio'] + [t for t in targets if t in merged.columns]
        f.write(merged[cols].sort_values('Fat_Ratio', ascending=False).head(15).to_markdown(index=False))

    logging.info(f"Analysis complete. Report saved to {REPORT_MD}")

def main():
    logging.info("Starting Task 16.3 V5.3: Cleaned Quantification")
    
    inv = get_inventory()
    if inv is None: return
    
    # Filter for valid paths
    targets = inv.dropna(subset=['Path', 'Patient_ID']).drop_duplicates(subset=['Patient_ID'])
    logging.info(f"Found {len(targets)} unique patients in inventory.")
    
    # Prepare task args
    tasks = []
    bad_ids = ["2020022863", "2020024526", "2120031865", "Exclude_Mouse_Data"]
    
    for _, row in targets.iterrows():
        pid = str(row['Patient_ID'])
        if pid in bad_ids or "Mouse" in pid:
            continue
        tasks.append((pid, row['Path']))        
    max_workers = max(1, os.cpu_count() - 2)
    logging.info(f"Processing with {max_workers} workers.")
    
    results = []
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_pid = {executor.submit(process_patient_worker, task): task[0] for task in tasks}
        for future in tqdm(concurrent.futures.as_completed(future_to_pid), total=len(tasks), desc="Quantifying V4"):
            results.append(future.result())

    duration = time.time() - start_time
    logging.info(f"Batch processing complete in {duration:.2f} seconds.")
    
    df_res = pd.DataFrame([r for r in results if r['Status'] == 'Success'])
    df_res.to_csv(OUTPUT_CSV, index=False)
    
    # Run Analysis
    run_analysis(df_res)

if __name__ == "__main__":
    main()
