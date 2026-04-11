import os
import shutil
import sqlite3
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import nibabel as nib
import numpy as np
from totalsegmentator.python_api import totalsegmentator
import logging
import sys
import json
import datetime
import traceback
import multiprocessing
import time

# Configuration
BASE_DIR = r"D:\mWork\paper0"
DB_PATH = os.path.join(BASE_DIR, "output", "task_3_cleaned.db")
CLEANED_DIR = os.path.join(BASE_DIR, "data", "CT_Cleaned")
TEMP_DIR = os.path.join(BASE_DIR, "data", "temp_processing")
OUTPUT_FEATURES_CSV = os.path.join(BASE_DIR, "output", "task_4_radiomics_features.csv")
OUTPUT_FAILED_CSV = os.path.join(BASE_DIR, "output", "task_4_failed_segmentations.csv")
REPORT_PATH = os.path.join(BASE_DIR, "output", "task_4_run_report.md")
LOG_PATH = os.path.join(BASE_DIR, "output", "task_4_log.txt")
CACHE_PROGRESS_FILE = os.path.join(BASE_DIR, "cache", "task_4_progress.json")

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FEATURES_CSV), exist_ok=True)
os.makedirs(os.path.dirname(CACHE_PROGRESS_FILE), exist_ok=True)

# Setup Logging (Main process only primarily, workers will inherit or need config)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def update_progress_safe(lock, status, processed_inc, total, success_inc, fail_inc):
    """Updates the progress file safely using a lock."""
    with lock:
        try:
            data = {}
            if os.path.exists(CACHE_PROGRESS_FILE):
                try:
                    with open(CACHE_PROGRESS_FILE, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except:
                    pass
            
            # Initialize if empty or new run
            if not data or status == "starting":
                data = {
                    "status": "starting",
                    "timestamp": str(datetime.datetime.now()),
                    "processed": 0,
                    "total": total,
                    "success": 0,
                    "failed": 0
                }
            
            data["status"] = status
            data["timestamp"] = str(datetime.datetime.now())
            data["processed"] = data.get("processed", 0) + processed_inc
            data["success"] = data.get("success", 0) + success_inc
            data["failed"] = data.get("failed", 0) + fail_inc
            
            # Only update total if it's the starting call
            if total > 0 and status == "starting":
                data["total"] = total

            with open(CACHE_PROGRESS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Progress update failed: {e}")

def init_radiomics_extractor():
    params = {}
    params['binWidth'] = 25
    params['resampledPixelSpacing'] = None 
    params['interpolator'] = sitk.sitkBSpline
    
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('gldm')
    extractor.enableFeatureClassByName('ngtdm')
    extractor.enableFeatureClassByName('shape') 
    
    return extractor

def process_single_patient(patient_id, gpu_id):
    # Unique temp dir for this process/patient to avoid collisions
    patient_temp_dir = os.path.join(TEMP_DIR, f"proc_{os.getpid()}_{patient_id}")
    os.makedirs(patient_temp_dir, exist_ok=True)
    
    patient_dir = os.path.join(CLEANED_DIR, patient_id)
    ct_nifti_path = os.path.join(patient_temp_dir, f"{patient_id}_ct.nii.gz")
    seg_output_dir = os.path.join(patient_temp_dir, f"{patient_id}_seg")
    liver_mask_path = os.path.join(seg_output_dir, "liver.nii.gz")
    
    extractor = init_radiomics_extractor()
    result = None
    
    try:
        # Step 1: DICOM to NIfTI
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(patient_dir)
        if not dicom_names:
            raise ValueError("No DICOM files found")
        
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        sitk.WriteImage(image, ct_nifti_path)
        
        # Step 2: TotalSegmentator
        # roi_subset=['liver'] for speed
        # Important: TotalSegmentator might use multiprocessing internally for some tasks, 
        # but the main heavy lifting is torch on GPU.
        # quiet=True to reduce log spam in parallel
        totalsegmentator(ct_nifti_path, seg_output_dir, roi_subset=['liver'], fast=True, ml=True, quiet=True)
        
        if not os.path.exists(liver_mask_path):
            raise FileNotFoundError("Liver mask not generated")
            
        # Step 3: Mask Validation
        mask_img = nib.load(liver_mask_path)
        mask_data = mask_img.get_fdata()
        if np.sum(mask_data) == 0:
            raise ValueError("Empty liver mask (no liver detected)")
            
        # Step 4: Feature Extraction
        feature_vector = extractor.execute(ct_nifti_path, liver_mask_path)
        
        result = dict(feature_vector)
        result['Patient_ID'] = patient_id
        
    except Exception as e:
        raise e
    finally:
        # Step 5: Cleanup
        if os.path.exists(patient_temp_dir):
            try:
                shutil.rmtree(patient_temp_dir, ignore_errors=True)
            except:
                pass
                
    return result

def worker_task(patient_list, gpu_id, lock):
    # Set CUDA device for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Reload torch/cuda context if needed (handled by os.environ usually if set before torch init)
    # Since this is 'spawn' on Windows, this is a fresh process. Perfect.
    
    local_results = []
    local_failures = []
    
    logging.info(f"Worker started on GPU {gpu_id} with {len(patient_list)} patients.")
    
    for pid in patient_list:
        logging.info(f"Processing {pid} on GPU {gpu_id}...")
        try:
            feats = process_single_patient(pid, gpu_id)
            local_results.append(feats)
            update_progress_safe(lock, "running", 1, 0, 1, 0)
        except Exception as e:
            err_msg = str(e)
            logging.error(f"Failed {pid} on GPU {gpu_id}: {err_msg}")
            local_failures.append({'Patient_ID': pid, 'Error': err_msg})
            update_progress_safe(lock, "running", 1, 0, 0, 1)
            
    return local_results, local_failures

def main():
    if not os.path.exists(DB_PATH):
        logging.error("Database not found.")
        return

    # 1. Get Patient List
    conn = sqlite3.connect(DB_PATH)
    patients = pd.read_sql_query("SELECT DISTINCT patient_id FROM cleaned_series", conn)['patient_id'].tolist()
    conn.close()
    
    total_patients = len(patients)
    logging.info(f"Total patients to process: {total_patients}")
    
    # 2. Configure Workers
    # GPU 0 (22GB): 2 workers
    # GPU 1 (12GB): 1 worker
    # Total 3 workers
    worker_configs = [
        {'gpu': 0, 'patients': []},
        {'gpu': 0, 'patients': []},
        {'gpu': 1, 'patients': []}
    ]
    
    # Distribute patients (Round Robin)
    for i, pid in enumerate(patients):
        worker_idx = i % len(worker_configs)
        worker_configs[worker_idx]['patients'].append(pid)
        
    # 3. Initialize Progress tracking
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    update_progress_safe(lock, "starting", 0, total_patients, 0, 0)
    
    # 4. Start Pool
    # We use a Pool but map arguments manually
    pool_args = []
    for cfg in worker_configs:
        if cfg['patients']: # Only start if has patients
            pool_args.append((cfg['patients'], cfg['gpu'], lock))
            
    start_time = datetime.datetime.now()
    
    with multiprocessing.Pool(processes=len(pool_args)) as pool:
        # starmap returns a list of (results, failures) tuples
        results_list = pool.starmap(worker_task, pool_args)
        
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    
    # 5. Aggregate Results
    all_features = []
    all_failures = []
    
    for res, fail in results_list:
        all_features.extend(res)
        all_failures.extend(fail)
        
    # 6. Save Outputs
    if all_features:
        df_res = pd.DataFrame(all_features)
        cols = list(df_res.columns)
        meta_cols = ['Patient_ID']
        diag_cols = [c for c in cols if c.startswith('diagnostics_')]
        feat_cols = [c for c in cols if c not in meta_cols and c not in diag_cols]
        df_res = df_res[meta_cols + feat_cols + diag_cols]
        
        df_res.to_csv(OUTPUT_FEATURES_CSV, index=False)
        logging.info(f"Features saved to {OUTPUT_FEATURES_CSV}")
        
    if all_failures:
        pd.DataFrame(all_failures).to_csv(OUTPUT_FAILED_CSV, index=False)
        logging.info(f"Failures saved to {OUTPUT_FAILED_CSV}")
        
    # 7. Final Report
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("# Task 4: Parallel Radiomics Extraction Report\n\n")
        f.write(f"Generated on: {datetime.datetime.now()}\n\n")
        f.write(f"* **Total Patients**: {total_patients}\n")
        f.write(f"* **Successfully Extracted**: {len(all_features)}\n")
        f.write(f"* **Failed**: {len(all_failures)}\n")
        f.write(f"* **Total Duration**: {duration}\n")
        f.write(f"* **Parallel Configuration**: GPU 0 (2 workers), GPU 1 (1 worker)\n\n")
        
        if all_failures:
            f.write("## Failed Patients\n")
            for fail in all_failures:
                f.write(f"* **{fail['Patient_ID']}**: {fail['Error']}\n")
                
    update_progress_safe(lock, "completed", 0, 0, 0, 0)
    logging.info("Task 4 Parallel Execution Completed.")

if __name__ == "__main__":
    # Support for PyInstaller/Windows multiprocessing
    multiprocessing.freeze_support()
    main()