import os
import shutil
import pandas as pd
import nibabel as nib
import numpy as np
import dicom2nifti
import logging
import sys
import datetime
import traceback
import multiprocessing
import torch
import subprocess
from multiprocessing import Manager, Semaphore

# --- Configuration ---
# CLI approach can be heavy, keep concurrency low
GPU_CAPACITIES = [1, 1] 
NUM_WORKERS = 2

BASE_DIR = "/mnt/d/mWork/paper0"
CLEANED_DIR = os.path.join(BASE_DIR, "data", "CT_Cleaned")
TEMP_BASE_DIR = os.path.join(BASE_DIR, "data", "temp_processing_task8_body")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FEATURES_CSV = os.path.join(OUTPUT_DIR, "task_8_body_features.csv")
FAILURES_CSV = os.path.join(OUTPUT_DIR, "task_8_body_failures.csv")
PREV_FEATURES = os.path.join(OUTPUT_DIR, "task_8_organ_features.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, "task_8_body_log.txt")
PROGRESS_TXT = os.path.join(BASE_DIR, "cache", "task_8_body_progress.txt")

os.makedirs(TEMP_BASE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def update_progress(progress_dict, total):
    try:
        processed = progress_dict.get("processed", 0)
        with open(PROGRESS_TXT, 'w', encoding='utf-8') as f:
            f.write(f"{processed}/{total}\n")
    except: pass

def get_main_ct_volume(temp_dir):
    nifti_files = [f for f in os.listdir(temp_dir) if f.endswith('.nii.gz') and not f.startswith('seg')]
    if not nifti_files: return None
    full_paths = [os.path.join(temp_dir, f) for f in nifti_files]
    full_paths.sort(key=lambda x: os.path.getsize(x), reverse=True)
    return full_paths[0]

def analyze_organ(ct_img, mask_img):
    try:
        mask_data = mask_img.get_fdata()
        if np.sum(mask_data) == 0: return np.nan, np.nan
        ct_data = ct_img.get_fdata()
        voxel_vol = np.prod(mask_img.header.get_zooms())
        volume_ml = (np.sum(mask_data > 0) * voxel_vol) / 1000.0
        mean_hu = ct_data[mask_data > 0].mean()
        return volume_ml, mean_hu
    except:
        return np.nan, np.nan

def process_patient(patient_id, gpu_id, gpu_semaphore):
    patient_dicom_dir = os.path.join(CLEANED_DIR, patient_id)
    patient_temp_dir = os.path.join(TEMP_BASE_DIR, f"tb_{os.getpid()}_{patient_id}")
    if os.path.exists(patient_temp_dir): shutil.rmtree(patient_temp_dir)
    os.makedirs(patient_temp_dir, exist_ok=True)
    
    try:
        # 1. DICOM -> NIfTI
        dicom2nifti.convert_directory(patient_dicom_dir, patient_temp_dir)
        ct_path = get_main_ct_volume(patient_temp_dir)
        if not ct_path: raise ValueError("DICOM fail")
        
        # 2. Segmentation (CLI - Using Tissue Types)
        seg_out_dir = os.path.join(patient_temp_dir, "seg")
        os.makedirs(seg_out_dir, exist_ok=True)
        
        logging.info(f"{patient_id}: Segmenting Tissue Types on GPU {gpu_id} (CLI)...")
        with gpu_semaphore:
            # Construct CLI command
            cmd = [
                "TotalSegmentator",
                "-i", ct_path,
                "-o", seg_out_dir,
                "-ta", "tissue_types",
                "-s", # Generate statistics.json
                "--quiet",
                "--device", f"gpu:{gpu_id}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"CLI Error: {result.stdout} {result.stderr}")
            
        # 3. Calculation (Read from NIfTI as existing logic works well)
        ct_img = nib.load(ct_path)
        results = {'Patient_ID': patient_id}
        
        # In tissue_types, the filenames are different
        target_rois = {
            'skeletal_muscle': 'Muscle',
            'subcutaneous_fat': 'Fat',
            'torso_fat': 'Visceral_Fat'
        }
        
        for ts_name, output_prefix in target_rois.items():
            mask_path = os.path.join(seg_out_dir, f"{ts_name}.nii.gz")
            if os.path.exists(mask_path):
                mask_img = nib.load(mask_path)
                vol, hu = analyze_organ(ct_img, mask_img)
                results[f'{output_prefix}_Volume'] = vol
                results[f'{output_prefix}_Mean_HU'] = hu
            else:
                results[f'{output_prefix}_Volume'] = np.nan
                results[f'{output_prefix}_Mean_HU'] = np.nan
            
        return results, None

    except Exception:
        return None, traceback.format_exc()
    finally:
        if os.path.exists(patient_temp_dir): shutil.rmtree(patient_temp_dir)

def worker_task(queue, gpu_id, semaphore, progress_dict, lock, total):
    while True:
        try: pid = queue.get_nowait() 
        except: break
        
        res, err = process_patient(pid, gpu_id, semaphore)
        with lock:
            progress_dict['processed'] += 1
            if res:
                df = pd.DataFrame([res])
                hdr = not os.path.exists(FEATURES_CSV)
                df.to_csv(FEATURES_CSV, mode='a', index=False, header=hdr)
            else:
                f_df = pd.DataFrame([{'Patient_ID': pid, 'Error': err, 'Time': str(datetime.datetime.now())}])
                f_hdr = not os.path.exists(FAILURES_CSV)
                f_df.to_csv(FAILURES_CSV, mode='a', index=False, header=f_hdr)
                logging.error(f"Failed {pid}")
            update_progress(progress_dict, total)

def main():
    multiprocessing.set_start_method('spawn', force=True)
    if not torch.cuda.is_available(): return

    if not os.path.exists(PREV_FEATURES):
        logging.error("Previous organ features not found.")
        return
        
    valid_pids = pd.read_csv(PREV_FEATURES)['Patient_ID'].astype(str).tolist()
    
    processed = set()
    if os.path.exists(FEATURES_CSV):
        try: processed.update(pd.read_csv(FEATURES_CSV)['Patient_ID'].astype(str).tolist())
        except: pass
    # Don't skip failures for retry this time, since we changed method
    # if os.path.exists(FAILURES_CSV):
    #    try: processed.update(pd.read_csv(FAILURES_CSV)['Patient_ID'].astype(str).tolist())
    #    except: pass
        
    to_process = [p for p in valid_pids if p not in processed]
    logging.info(f"Targeting {len(to_process)} remaining patients.")
    
    if not to_process: return

    q = multiprocessing.Queue()
    for p in to_process: q.put(p)
    
    mgr = Manager()
    prog = mgr.dict()
    prog['processed'] = 0
    lock = mgr.Lock()
    
    semaphores = [Semaphore(cap) for cap in GPU_CAPACITIES]
    procs = []
    assignments = [0, 1] 
    
    for i in range(len(assignments)):
        gpu = assignments[i]
        p = multiprocessing.Process(target=worker_task, args=(q, gpu, semaphores[gpu], prog, lock, len(to_process)))
        p.start()
        procs.append(p)
        
    for p in procs: p.join()
    logging.info("Done.")

if __name__ == "__main__":
    main()
