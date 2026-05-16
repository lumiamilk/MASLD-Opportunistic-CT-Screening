import os
import shutil
import sqlite3
import pandas as pd
import pydicom
import logging
import sys
from tqdm import tqdm

# Configuration
BASE_DIR = r"D:\\mWork\\paper0"
PLAN_CSV = os.path.join(BASE_DIR, "output", "ct_selection_plan.csv")
TARGET_DIR = os.path.join(BASE_DIR, "data", "CT_Cleaned")
DB_PATH = os.path.join(BASE_DIR, "output", "task_3_cleaned.db")
REPORT_PATH = os.path.join(BASE_DIR, "output", "task_3_run_report.md")
LOG_PATH = os.path.join(BASE_DIR, "output", "task_3_log.txt")

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS cleaned_series")
    cursor.execute("""
        CREATE TABLE cleaned_series (
            patient_id TEXT,
            series_uid TEXT,
            original_path TEXT,
            cleaned_path TEXT,
            series_description TEXT,
            kernel TEXT,
            slice_thickness REAL
        )
    """)
    conn.commit()
    conn.close()

def main():
    if not os.path.exists(PLAN_CSV):
        logging.error(f"Plan CSV not found: {PLAN_CSV}")
        return

    # Create target directory
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Init DB
    init_db()
    conn = sqlite3.connect(DB_PATH)
    
    # Read Plan
    df = pd.read_csv(PLAN_CSV)
    logging.info(f"Loaded selection plan: {len(df)} patients.")
    
    stats = {
        "processed_patients": 0,
        "files_hardlinked": 0,
        "files_copied": 0,
        "uid_mismatch_skipped": 0,
        "errors": 0,
        "space_saved_bytes": 0
    }
    
    # Iterate with TQDM
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Patients"):
        stats["processed_patients"] += 1
        
        patient_id = str(row['Patient_ID'])
        target_uid = str(row['Selected_Series_UID'])
        src_folder = row['Folder_Path']
        
        # Validation
        if pd.isna(src_folder) or not os.path.exists(src_folder):
            logging.warning(f"Source folder not found for Patient {patient_id}: {src_folder}")
            stats["errors"] += 1
            continue
            
        # Target Patient Folder
        patient_clean_dir = os.path.join(TARGET_DIR, patient_id)
        os.makedirs(patient_clean_dir, exist_ok=True)
        
        # Scan files in source folder
        # Note: os.walk might be overkill if we just need the flat folder listed in plan, 
        # but safely handles subdirs if present.
        for root, _, files in os.walk(src_folder):
            for file in files:
                src_path = os.path.join(root, file)
                
                try:
                    # 1. Validation Step: Read DICOM UID
                    try:
                        # force=True is important for files missing preamble
                        ds = pydicom.dcmread(src_path, stop_before_pixels=True, force=True)
                        file_uid = str(ds.get("SeriesInstanceUID", ""))
                        
                        if file_uid != target_uid:
                            stats["uid_mismatch_skipped"] += 1
                            continue # Skip files not belonging to selected series
                            
                    except Exception:
                        # Not a DICOM file or unreadable
                        continue
                        
                    # 2. Link/Copy Step
                    dst_path = os.path.join(patient_clean_dir, file)
                    
                    if os.path.exists(dst_path):
                        # Skip if already done (idempotency)
                        continue
                        
                    file_size = os.path.getsize(src_path)
                    
                    try:
                        os.link(src_path, dst_path)
                        stats["files_hardlinked"] += 1
                        stats["space_saved_bytes"] += file_size
                    except OSError:
                        # Fallback to copy (e.g., cross-drive)
                        shutil.copy2(src_path, dst_path)
                        stats["files_copied"] += 1
                        logging.warning(f"Fallback to copy for {src_path}")
                        
                    # 3. DB Insert
                    conn.execute("""
                        INSERT INTO cleaned_series 
                        (patient_id, series_uid, original_path, cleaned_path, series_description, kernel, slice_thickness)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        patient_id, 
                        target_uid, 
                        src_path, 
                        dst_path, 
                        row['Series_Description'], 
                        row['Kernel'], 
                        row['Slice_Thickness']
                    ))
                    
                except Exception as e:
                    logging.error(f"Error processing file {src_path}: {e}")
                    stats["errors"] += 1

        conn.commit()
    
    conn.close()
    
    # Generate Report
    generate_report(stats)

def generate_report(stats):
    space_mb = stats['space_saved_bytes'] / (1024 * 1024)
    space_gb = space_mb / 1024
    
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("# Task 3: CT Data Cleaning & Restructuring Report\n\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
        
        f.write("## 1. Executive Summary\n")
        f.write(f"* **Processed Patients**: {stats['processed_patients']}\n")
        f.write(f"* **Total Files Structured**: {stats['files_hardlinked'] + stats['files_copied']}\n")
        f.write(f"* **Hard Links Created**: {stats['files_hardlinked']} (Instant, Zero Space)\n")
        f.write(f"* **Fallback Copies**: {stats['files_copied']}\n")
        f.write(f"* **Est. Space Saved**: {space_mb:.2f} MB ({space_gb:.2f} GB)\n\n")
        
        f.write("## 2. Integrity Checks\n")
        f.write(f"* **UID Mismatches Skipped**: {stats['uid_mismatch_skipped']}\n")
        f.write("  *(Files found in source folders that belonged to other series)*\n")
        f.write(f"* **Errors Encountered**: {stats['errors']}\n\n")
        
        f.write("## 3. Output Location\n")
        f.write(f"* **Cleaned Data**: `{TARGET_DIR}`\n")
        f.write(f"* **Tracking DB**: `{DB_PATH}`\n")

    logging.info(f"Task 3 Complete. Report at {REPORT_PATH}")
    # Print report to stdout for the agent to see
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        print(f.read())

if __name__ == "__main__":
    main()
