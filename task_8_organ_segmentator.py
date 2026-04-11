import os
import shutil
import pandas as pd
import nibabel as nib
import numpy as np
import totalsegmentator.python_api as ts_api
import dicom2nifti
import logging
import sys
import datetime
import traceback
import multiprocessing
import torch
from multiprocessing import Manager, Semaphore

# --- Configuration ---
# 165 patients, similar to Task 4 setup
GPU_CAPACITIES = [2, 1] 
NUM_WORKERS = 3

BASE_DIR = "/mnt/d/mWork/paper0"
CLEANED_DIR = os.path.join(BASE_DIR, "data", "CT_Cleaned")
TEMP_BASE_DIR = os.path.join(BASE_DIR, "data", "temp_processing_task8")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FEATURES_CSV = os.path.join(OUTPUT_DIR, "task_8_organ_features.csv")
FAILURES_CSV = os.path.join(OUTPUT_DIR, "task_8_failures.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, "task_8_log.txt")
PROGRESS_TXT = os.path.join(BASE_DIR, "cache", "task_8_progress.txt")

os.makedirs(TEMP_BASE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PROGRESS_TXT), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def update_progress(progress_dict, total_patients):
    try:
        start_time = progress_dict.get("start_time")
        elapsed_str = "00:00:00"
        if start_time:
            import time
            elapsed = time.time() - start_time
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
            
        processed = progress_dict.get("processed", 0)
        with open(PROGRESS_TXT, 'w', encoding='utf-8') as f:
            f.write(f"{processed}/{total_patients} - Elapsed: {elapsed_str}\n")
    except:
        pass

def get_main_ct_volume(temp_dir):
    # Same logic as Task 4
    nifti_files = [f for f in os.listdir(temp_dir) if f.endswith('.nii.gz') and not f.startswith('seg')]
    if not nifti_files: return None
    full_paths = [os.path.join(temp_dir, f) for f in nifti_files]
    full_paths.sort(key=lambda x: os.path.getsize(x), reverse=True)
    return full_paths[0]

def analyze_organ(ct_img, mask_img, label_name):
    """Calculates Volume (mL) and Mean HU."""
    try:
        mask_data = mask_img.get_fdata()
        if np.sum(mask_data) == 0:
            return np.nan, np.nan
            
        ct_data = ct_img.get_fdata()
        
        # Volume
        voxel_vol = np.prod(mask_img.header.get_zooms())
        volume_ml = (np.sum(mask_data > 0) * voxel_vol) / 1000.0
        
        # Mean HU
        # Apply mask to CT
        mean_hu = ct_data[mask_data > 0].mean()
        
        return volume_ml, mean_hu
    except Exception as e:
        logging.error(f"Error analyzing {label_name}: {e}")
        return np.nan, np.nan

def process_patient(patient_id, gpu_id, gpu_semaphore):
    patient_dicom_dir = os.path.join(CLEANED_DIR, patient_id)
    # Check if cleaned dir exists (it is now the ID itself)
    if not os.path.exists(patient_dicom_dir):
        return None, f"Folder not found: {patient_dicom_dir}"

    patient_temp_dir = os.path.join(TEMP_BASE_DIR, f"t8_{os.getpid()}_{patient_id}")
    if os.path.exists(patient_temp_dir): shutil.rmtree(patient_temp_dir)
    os.makedirs(patient_temp_dir, exist_ok=True)
    
    try:
        # 1. DICOM -> NIfTI
        logging.info(f"{patient_id}: Converting DICOM...")
        dicom2nifti.convert_directory(patient_dicom_dir, patient_temp_dir)
        ct_path = get_main_ct_volume(patient_temp_dir)
        if not ct_path: raise ValueError("DICOM conversion failed")
        
        # 2. Segmentation
        seg_out_dir = os.path.join(patient_temp_dir, "seg")
        os.makedirs(seg_out_dir, exist_ok=True)
        
        # Simplified ROIs for stability (liver and spleen only)
        rois = ['liver', 'spleen']
        
        logging.info(f"{patient_id}: Segmenting on GPU {gpu_id}...")
        with gpu_semaphore:
            ts_api.totalsegmentator(
                ct_path, seg_out_dir, roi_subset=rois,
                ml=False, fast=True, quiet=True, device=f"gpu:{gpu_id}"
            )
            
        # 3. Calculation
        ct_img = nib.load(ct_path)
        results = {'Patient_ID': patient_id}
        
        for roi in rois:
            mask_path = os.path.join(seg_out_dir, f"{roi}.nii.gz")
            if os.path.exists(mask_path):
                mask_img = nib.load(mask_path)
                vol, hu = analyze_organ(ct_img, mask_img, roi)
            else:
                vol, hu = np.nan, np.nan
            
            # Map names to output format
            prefix = roi.title().replace('Skeletal_', '').replace('Subcutaneous_', '') # Liver, Spleen, Muscle, Fat
            results[f'{prefix}_Volume'] = vol
            if roi != 'subcutaneous_fat': # Fat HU is constant (-100ish), less useful
                results[f'{prefix}_Mean_HU'] = hu
                
        # 4. Derived Ratios
        if not pd.isna(results.get('Liver_Mean_HU')) and not pd.isna(results.get('Spleen_Mean_HU')) and results['Spleen_Mean_HU'] != 0:
            results['L_S_Ratio'] = results['Liver_Mean_HU'] / results['Spleen_Mean_HU']
        else:
            results['L_S_Ratio'] = np.nan
            
        return results, None

    except Exception:
        return None, traceback.format_exc()
    finally:
        if os.path.exists(patient_temp_dir): shutil.rmtree(patient_temp_dir)

def worker_task(queue, gpu_id, semaphore, progress_dict, lock, total):
    while True:
        try:
            pid = queue.get_nowait()
        except:
            break
            
        logging.info(f"Worker GPU {gpu_id} processing {pid}")
        res, err = process_patient(pid, gpu_id, semaphore)
        
        with lock:
            progress_dict['processed'] += 1
            if res:
                df = pd.DataFrame([res])
                hdr = not os.path.exists(FEATURES_CSV)
                df.to_csv(FEATURES_CSV, mode='a', index=False, header=hdr)
            else:
                f_df = pd.DataFrame([{'Patient_ID': pid, 'Error': err}])
                hdr = not os.path.exists(FAILURES_CSV)
                f_df.to_csv(FAILURES_CSV, mode='a', index=False, header=hdr)
            
            update_progress(progress_dict, total)

def main():
    import time
    multiprocessing.set_start_method('spawn', force=True)
    if not torch.cuda.is_available():
        logging.error("No GPU.")
        return

    # Use MFT Mapping to find Real IDs if needed, but here we scan directories directly
    # The folders are now renamed to Real IDs (Task 6), so os.listdir is correct.
    all_pids = [d for d in os.listdir(CLEANED_DIR) if os.path.isdir(os.path.join(CLEANED_DIR, d))]
    
    # Resume
    processed = set()
    if os.path.exists(FEATURES_CSV):
        try: processed.update(pd.read_csv(FEATURES_CSV)['Patient_ID'].astype(str).tolist())
        except: pass
        
    to_process = [p for p in all_pids if p not in processed]
    
    logging.info(f"Total: {len(all_pids)}, Remaining: {len(to_process)}")
    
    if not to_process: return

    # --- Warmup / Weight Download ---
    # We must run one segmentation in the main process first to ensure weights are downloaded.
    # Otherwise, parallel workers will race to download and crash.
    logging.info("Starting warmup (weight download)...")
    try:
        # Dummy run on the first patient
        warmup_pid = to_process[0]
        # We don't process it fully, just trigger the download logic
        # But easier to just process one patient fully in main thread
        # Note: We use GPU 0 for warmup
        sem = Semaphore(1)
        logging.info(f"Warming up with {warmup_pid}")
        res, err = process_patient(warmup_pid, 0, sem)
        if res:
            logging.info("Warmup successful.")
            # Add to processed manually so workers don't redo it
            processed.add(warmup_pid)
            df = pd.DataFrame([res])
            hdr = not os.path.exists(FEATURES_CSV)
            df.to_csv(FEATURES_CSV, mode='a', index=False, header=hdr)
            to_process = to_process[1:] # Remove from queue
        else:
            logging.error(f"Warmup failed: {err}")
            # If warmup fails, parallel will likely fail too, but we proceed
    except Exception as e:
        logging.error(f"Warmup crashed: {e}")

    if not to_process:
        logging.info("All done after warmup.")
        return

    q = multiprocessing.Queue()
    for p in to_process: q.put(p)
    
    mgr = Manager()
    prog = mgr.dict()
    prog['processed'] = 0
    prog['start_time'] = time.time()
    lock = mgr.Lock()
    
    semaphores = [Semaphore(cap) for cap in GPU_CAPACITIES]
    
    procs = []
    # Distribute: GPU0 gets 2/3, GPU1 gets 1/3
    assignments = [0, 0, 1] 
    
    for i in range(NUM_WORKERS):
        gpu = assignments[i]
        p = multiprocessing.Process(target=worker_task, args=(q, gpu, semaphores[gpu], prog, lock, len(to_process)))
        p.start()
        procs.append(p)
        
    for p in procs: p.join()
    logging.info("Done.")

if __name__ == "__main__":
    main()
