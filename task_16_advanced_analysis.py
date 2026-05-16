import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import SimpleITK as sitk
import openslide
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import logging
import warnings
import concurrent.futures
import scipy.ndimage as ndimage

# --- Config ---
BASE_DIR = "/mnt/d/mWork/paper0"
RESULT_CSV = os.path.join(BASE_DIR, "output", "task_16_quant_results_v5.3.csv")
MASTER_CSV = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
INVENTORY_CSV = os.path.join(BASE_DIR, "output", "task_16_pathology_inventory.csv")
RETICULIN_DIR = os.path.join(BASE_DIR, "data", "临床liver 切片", "临床 liver 网染")
CT_CLEANED_DIR = os.path.join(BASE_DIR, "data", "CT_Cleaned")
OUTPUT_REPORT = os.path.join(BASE_DIR, "output", "task_16_advanced_report.md")
OUTPUT_FIG_GRADE = os.path.join(BASE_DIR, "output", "patho_grade_validation.png")
OUTPUT_FIG_CASE = os.path.join(BASE_DIR, "output", "reticulin_case_study.png")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
warnings.filterwarnings("ignore")

def load_data():
    if not os.path.exists(RESULT_CSV):
        raise FileNotFoundError(f"Result CSV not found: {RESULT_CSV}")
    
    df_res = pd.read_csv(RESULT_CSV, dtype={'Patient_ID': str})
    df_master = pd.read_csv(MASTER_CSV, dtype={'patient_id': str})
    df_master['patient_id'] = df_master['patient_id'].str.strip()
    
    merged = pd.merge(df_res, df_master, left_on='Patient_ID', right_on='patient_id', how='inner')
    return merged

def analyze_grading(merged):
    print("Analyzing Fat Ratio vs Steatosis Grade...", flush=True)
    valid = merged.dropna(subset=['Steatosis_Grade', 'Fat_Ratio'])
    valid['Steatosis_Grade'] = valid['Steatosis_Grade'].astype(int)
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=valid, x='Steatosis_Grade', y='Fat_Ratio', palette="Blues")
    sns.stripplot(data=valid, x='Steatosis_Grade', y='Fat_Ratio', color='black', alpha=0.5)
    
    if len(valid) > 1:
        r, p_spearman = stats.spearmanr(valid['Steatosis_Grade'], valid['Fat_Ratio'])
        groups = [valid[valid['Steatosis_Grade'] == g]['Fat_Ratio'] for g in sorted(valid['Steatosis_Grade'].unique())]
        if len(groups) > 1:
            stat, p_kw = stats.kruskal(*groups)
        else:
            stat, p_kw = 0, 1.0
    else:
        r, p_spearman, p_kw = 0, 1.0, 1.0

    plt.title(f"Fat Ratio vs Steatosis Grade\nN={len(valid)}, Spearman R={r:.3f}, KW p={p_kw:.3e}")
    plt.xlabel("Pathologist Grade (0-3)")
    plt.ylabel("AI Fat Ratio")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_GRADE)
    plt.close()
    return valid, r, p_spearman, p_kw

def find_best_liver_slice(ct_path):
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(ct_path)
        if not dicom_names: return None
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        arr = sitk.GetArrayFromImage(image)
        z_depth = arr.shape[0]
        
        max_liver_pixels = 0
        best_slice_idx = 0
        stride = 3
        
        # Simple heuristic to find liver: pixels in [30, 150] HU
        for z in range(0, z_depth, stride):
            slice_data = arr[z, :, :]
            mask = (slice_data >= 30) & (slice_data <= 150)
            area = np.sum(mask)
            if area > max_liver_pixels:
                max_liver_pixels = area
                best_slice_idx = z
                
        slice_img = arr[best_slice_idx, :, :]
        # Windowing (Level 40, Width 350)
        level, width = 40, 350
        min_val = level - width / 2
        max_val = level + width / 2
        slice_img = np.clip(slice_img, min_val, max_val)
        slice_img = ((slice_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return slice_img
    except Exception as e:
        print(f"CT Slice Error: {e}", flush=True)
        return None

def find_reticulin_file(patient_id, reticulin_files_cache):
    inv = pd.read_csv(INVENTORY_CSV, dtype=str)
    row = inv[inv['Patient_ID'] == patient_id]
    if row.empty: return None
    
    path_id = row.iloc[0]['Extracted_PathID']
    possible_ids = [str(path_id)]
    if str(path_id).startswith('L'):
        possible_ids.append(path_id[1:]) 
    
    for f in reticulin_files_cache:
        for pid in possible_ids:
            # Match strict ID to avoid partial matches
            if pid in f:
                full_path = os.path.join(RETICULIN_DIR, f)
                if os.path.isfile(full_path) and f.lower().endswith(('.mrxs', '.svs', '.ndpi', '.tif')):
                    return full_path
    return None

def get_slide_scan_info(path, mode='density'):
    """
    Scans the slide and returns the BEST ROI coordinates and a validity score.
    Uses a sliding window approach on a thumbnail to maximize tissue content
    for a target 2048x2048 crop.
    """
    slide = None
    try:
        slide = openslide.OpenSlide(path)
        w, h = slide.dimensions
        
        # 1. Create a larger thumbnail for better precision (Target ~2048px dim)
        #    This makes 1 pixel in thumbnail approx 10-40 pixels in full res
        target_thumb_dim = 2048
        downsample_factor = max(w, h) / target_thumb_dim
        thumb_w = int(w / downsample_factor)
        thumb_h = int(h / downsample_factor)
        
        thumb = slide.get_thumbnail((thumb_w, thumb_h))
        arr_thumb = np.array(thumb)
        gray = rgb2gray(arr_thumb)
        
        # 2. Robust Tissue Masking
        if arr_thumb.shape[2] == 4:
            alpha = arr_thumb[:, :, 3]
            mask = alpha > 10
        else:
            # Simple brightness threshold (exclude white background)
            # Relaxed to 0.90 to catch fainter H&E
            mask = (gray < 0.90) & (gray > 0.05)
            
        # Clean up small noise
        mask = ndimage.binary_opening(mask, structure=np.ones((3,3)))
        
        # 3. Define the Sliding Window Size
        #    We want a 2048x2048 box in Level 0. 
        #    In thumbnail pixels, this is:
        box_size_thumb = 2048.0 / downsample_factor
        k_size = int(round(box_size_thumb))
        
        # Safety check: if slide is smaller than 2048, take whole slide
        if k_size > min(thumb_w, thumb_h):
             k_size = min(thumb_w, thumb_h)
             
        if k_size < 1: k_size = 1

        # 4. Convolve to find region with MOST tissue
        #    We use a uniform filter which computes the mean.
        #    Maximizing mean is same as maximizing sum.
        metric = mask.astype(float)
        
        # If fibrosis mode, we might prefer darker (denser) areas, 
        # but primarily we just want TISSUE first to avoid white gaps.
        # We can weigh darker pixels slightly higher.
        if mode == 'fibrosis':
             # Weighted: 70% existence, 30% darkness
             weight_map = mask.astype(float) * 0.7 + (mask * (1.0 - gray)) * 0.3
             density_map = ndimage.uniform_filter(weight_map, size=k_size, mode='constant', cval=0.0)
        else:
             density_map = ndimage.uniform_filter(metric, size=k_size, mode='constant', cval=0.0)
        
        # 5. Find Peak
        flat_idx = np.argmax(density_map)
        cy, cx = np.unravel_index(flat_idx, density_map.shape)
        max_val = density_map[cy, cx] # This is 0.0 to 1.0 (mean coverage)
        
        # 6. Map center back to Top-Left in Level 0
        #    'cy, cx' is the center of the window in thumbnail space
        real_cx = cx * downsample_factor
        real_cy = cy * downsample_factor
        
        top_left_x = int(real_cx - 1024)
        top_left_y = int(real_cy - 1024)
        
        # Clamp to bounds
        top_left_x = max(0, min(w - 2048, top_left_x))
        top_left_y = max(0, min(h - 2048, top_left_y))
        
        # 7. Check Validity (Coverage)
        #    If the best window has < 20% tissue, it's garbage.
        if max_val < 0.20:
            return False, max_val, (0,0)
            
        return True, max_val, (top_left_x, top_left_y)
        
    except Exception as e:
        # print(f"Scan Error {os.path.basename(path)}: {e}", flush=True)
        return False, 0, (0,0)
    finally:
        if slide: slide.close()

def get_full_resolution_roi(path, top_left, size=2048):
    slide = openslide.OpenSlide(path)
    try:
        roi = slide.read_region(top_left, 0, (size, size)).convert("RGB")
        return np.array(roi)
    finally:
        slide.close()

def process_reticulin(img_arr):
    gray = rgb2gray(img_arr)
    # Adaptive threshold
    try:
        thresh = threshold_otsu(gray)
        binary = gray < thresh 
        overlay = img_arr.copy()
        # Cyan highlight
        overlay[binary] = [0, 255, 255]
        return overlay
    except:
        return img_arr

# --- Worker Function ---
def evaluate_patient(args):
    """
    Worker function to check if a patient is a good candidate.
    Returns metadata, NOT images.
    """
    pid, ret_files_cache, inv_csv_path, ct_dir, ret_dir = args
    
    # 1. CT Check
    ct_path = os.path.join(ct_dir, pid)
    if not os.path.exists(ct_path):
        return None
    
    # 2. Inventory Check (Reload inside worker to avoid pickle issues, it's fast)
    inv = pd.read_csv(inv_csv_path)
    he_rows = inv[inv['Patient_ID'] == pid]
    if he_rows.empty:
        return None
    he_path = he_rows.iloc[0]['Path']
    
    # 3. Reticulin File Check
    # Helper duplicated or imported? Let's just duplicate logic briefly or use static
    # Re-implement simple find_reticulin logic here to be self-contained or importable
    # We will assume 'find_reticulin_file' is available in global scope of worker if using fork
    # But safer to copy logic
    
    path_id = he_rows.iloc[0]['Extracted_PathID']
    possible_ids = [str(path_id)]
    if str(path_id).startswith('L'): possible_ids.append(path_id[1:])
    
    r_path = None
    for f in ret_files_cache:
        for xid in possible_ids:
            if xid in f:
                fp = os.path.join(ret_dir, f)
                if os.path.isfile(fp) and f.lower().endswith(('.mrxs', '.svs', '.ndpi', '.tif')):
                    r_path = fp
                    break
        if r_path: break
    
    if not r_path: return None
    
    # 4. Scan Slides (Lightweight)
    he_valid, he_score, he_coords = get_slide_scan_info(he_path, mode='density')
    if not he_valid or he_score < 0.5: return None
    
    ret_valid, ret_score, ret_coords = get_slide_scan_info(r_path, mode='fibrosis')
    if not ret_valid or ret_score < 0.5: return None
    
    return {
        'pid': pid,
        'he_path': he_path,
        'he_coords': he_coords,
        'ret_path': r_path,
        'ret_coords': ret_coords,
        'score': he_score + ret_score
    }

def case_study(merged):
    print("Searching for valid Case Study (Lightweight Scan)...", flush=True)
    # Prioritize high fat grade for better visual
    candidates = merged.sort_values('Steatosis_Grade', ascending=False)
    
    try:
        ret_files_cache = os.listdir(RETICULIN_DIR)
    except Exception as e:
        print(f"Error accessing reticulin dir: {e}")
        return
        
    # Prepare task arguments
    # Check top 40 candidates
    tasks = []
    for _, row in candidates.head(40).iterrows():
        tasks.append((row['Patient_ID'], ret_files_cache, INVENTORY_CSV, CT_CLEANED_DIR, RETICULIN_DIR))
        
    best_candidate = None
    
    # Use max 4 workers to save memory
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(evaluate_patient, t): t[0] for t in tasks}
        
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            pid = futures[future]
            try:
                res = future.result()
                if res:
                    print(f"  [Hit] {pid} is valid (Score: {res['score']:.2f})", flush=True)
                    if best_candidate is None or res['score'] > best_candidate['score']:
                        best_candidate = res
                        # Early exit if we find a very good one
                        if res['score'] > 1.8: 
                            print("  Found excellent candidate. Stopping search.")
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                completed_count += 1
                if completed_count % 5 == 0:
                    print(f"  Scanned {completed_count} candidates...", flush=True)
            except Exception as e:
                print(f"  [Err] {pid}: {e}", flush=True)
                
    if not best_candidate:
        print("No suitable case found.")
        return

    print(f"\nGenerating visualization for Best Candidate: {best_candidate['pid']}")
    
    # Now load the heavy images in the main process
    try:
        ct_img = find_best_liver_slice(os.path.join(CT_CLEANED_DIR, best_candidate['pid']))
        he_roi = get_full_resolution_roi(best_candidate['he_path'], best_candidate['he_coords']).astype(np.uint8)
        ret_roi = get_full_resolution_roi(best_candidate['ret_path'], best_candidate['ret_coords']).astype(np.uint8)
        ret_vis = process_reticulin(ret_roi)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        
        if ct_img is not None:
            axes[0].imshow(ct_img, cmap='gray')
            axes[0].set_title(f"CT (Liver)\n{best_candidate['pid']}", fontsize=14)
        else:
            axes[0].text(0.5, 0.5, "CT Missing", ha='center')
            
        axes[1].imshow(he_roi)
        axes[1].set_title("H&E (Fat)", fontsize=14)
        
        axes[2].imshow(ret_vis)
        axes[2].set_title("Reticulin (Fibrosis)", fontsize=14)
        
        for ax in axes: ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_FIG_CASE)
        print(f"Saved: {OUTPUT_FIG_CASE}")
        plt.close()
        
    except Exception as e:
        print(f"Error generating final figure: {e}")

def main():
    print("Script started.", flush=True)
    try:
        merged = load_data()
        valid_df, r, p_s, p_kw = analyze_grading(merged)
        
        # Write report
        with open(OUTPUT_REPORT, 'w') as f:
            f.write("# Task 16 Advanced Analysis Report\n\n")
            f.write("## 1. Steatosis Grading Validation\n")
            f.write(f"- **Sample Size**: {len(valid_df)}\n")
            f.write(f"- **Spearman Correlation**: {r:.3f} (p={p_s:.3e})\n")
            f.write(f"- **Kruskal-Wallis Test**: p={p_kw:.3e}\n")
            f.write("![Boxplot](patho_grade_validation.png)\n\n")
            f.write("## 2. Reticulin Visualization\n")
            f.write("See `reticulin_case_study.png` for multi-modal comparison.\n")
            
        case_study(merged)
        print("All tasks completed.")
        
    except Exception as e:
        print(f"Main Execution Error: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()