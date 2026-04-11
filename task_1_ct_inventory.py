import os
import sqlite3
import pydicom
import logging
import datetime
import sys

import json

# Configuration
# Using absolute paths based on the project root D:\mWork\paper0
BASE_DIR = r"D:\mWork\paper0"
DATA_DIR = os.path.join(BASE_DIR, "data", "CT_origianl_data_2018_2025")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
DB_PATH = os.path.join(OUTPUT_DIR, "task_1_ct_metadata.db")
REPORT_PATH = os.path.join(OUTPUT_DIR, "task_1_run_report.md")
LOG_PATH = os.path.join(OUTPUT_DIR, "task_1_log.txt")
PROGRESS_FILE = os.path.join(CACHE_DIR, "task_1_progress.json")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def update_progress(status, total_files, dicom_files, series_count):
    """Updates the progress file."""
    try:
        data = {
            "status": status,
            "timestamp": str(datetime.datetime.now()),
            "total_files_scanned": total_files,
            "valid_dicom_files": dicom_files,
            "unique_series_found": series_count
        }
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass # Don't crash on progress update fail

def init_db(cursor):
    """Initialize the SQLite database with the required schema."""
    cursor.execute('DROP TABLE IF EXISTS ct_series')
    cursor.execute('''
        CREATE TABLE ct_series (
            series_uid TEXT PRIMARY KEY,
            patient_id TEXT,
            study_date TEXT,
            series_description TEXT,
            modality TEXT,
            kernel TEXT,
            slice_thickness REAL,
            kvp REAL,
            image_count INTEGER,
            file_path_sample TEXT
        )
    ''')

def get_tag_value(dataset, tag, default=None):
    """Safely get a tag value from a pydicom dataset."""
    if tag not in dataset:
        return default
    val = dataset.get(tag)
    if val is None:
        return default
    if isinstance(val, pydicom.multival.MultiValue):
        # Join multi-values with a slash
        return "/".join([str(v) for v in val])
    return str(val)

def clean_id(raw_id):
    """Standardize ID: remove non-alphanumeric characters."""
    if not raw_id:
        return "UNKNOWN"
    return "".join(c for c in str(raw_id) if c.isalnum())

def scan_and_populate(cursor, data_dir):
    """Recursively scan the directory and populate the database."""
    series_registry = {}
    total_files = 0
    dicom_files = 0
    
    logging.info(f"Starting scan of {data_dir}...")
    
    for root, dirs, files in os.walk(data_dir):
        # Extract potential Patient ID from directory name if it looks like one
        # This is a fallback if the DICOM tag is missing
        dir_name = os.path.basename(root)
        potential_pid = clean_id(dir_name) if dir_name.isdigit() else None
        
        for file in files:
            total_files += 1
            file_path = os.path.join(root, file)
            
            try:
                # Read dataset, stopping before pixels to save time/memory
                ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
                
                # Check for SeriesInstanceUID to confirm it's a relevant DICOM file
                if "SeriesInstanceUID" not in ds:
                    continue
                
                dicom_files += 1
                series_uid = str(ds.SeriesInstanceUID)
                
                if series_uid in series_registry:
                    series_registry[series_uid]['image_count'] += 1
                else:
                    # New series found, extract metadata
                    
                    # 1. Patient ID
                    raw_pid = get_tag_value(ds, "PatientID")
                    patient_id = clean_id(raw_pid)
                    if (not patient_id or patient_id == "UNKNOWN") and potential_pid:
                        patient_id = potential_pid
                    
                    # 2. Modality
                    modality = get_tag_value(ds, "Modality", "UNKNOWN")
                    
                    # 3. Kernel (Merge Convolution and Reconstruction Kernel)
                    conv_kernel = get_tag_value(ds, "ConvolutionKernel", "")
                    recon_kernel = get_tag_value(ds, "ReconstructionKernel", "")
                    # Prefer non-empty
                    kernel = conv_kernel if conv_kernel else recon_kernel
                    if not kernel:
                        kernel = ""
                    
                    # 4. Description
                    desc = get_tag_value(ds, "SeriesDescription", "").strip()
                    
                    # 5. Study Date
                    study_date = get_tag_value(ds, "StudyDate", "")
                    
                    # 6. Technical Params
                    try:
                        slice_thickness = float(ds.SliceThickness) if "SliceThickness" in ds else None
                    except:
                        slice_thickness = None
                        
                    try:
                        kvp = float(ds.KVP) if "KVP" in ds else None
                    except:
                        kvp = None

                    series_registry[series_uid] = {
                        'series_uid': series_uid,
                        'patient_id': patient_id,
                        'study_date': study_date,
                        'series_description': desc,
                        'modality': modality,
                        'kernel': kernel,
                        'slice_thickness': slice_thickness,
                        'kvp': kvp,
                        'image_count': 1,
                        'file_path_sample': file_path
                    }
                    
            except Exception as e:
                # Log specific errors if needed, but for bulk scan usually we just skip
                # logging.debug(f"Skipping {file_path}: {e}")
                pass
                
            if total_files % 100 == 0:
                print(f"Scanned {total_files} files... (DICOMs: {dicom_files})", end='\r')
                update_progress("running", total_files, dicom_files, len(series_registry))

    print(f"\nScan complete. Found {len(series_registry)} unique series from {dicom_files} DICOM files.")
    logging.info(f"Scan complete. Found {len(series_registry)} unique series from {dicom_files} valid DICOM files out of {total_files} total files.")
    update_progress("processing_db", total_files, dicom_files, len(series_registry))
    
    # Batch Insert
    rows = []
    for uid, data in series_registry.items():
        rows.append((
            data['series_uid'],
            data['patient_id'],
            data['study_date'],
            data['series_description'],
            data['modality'],
            data['kernel'],
            data['slice_thickness'],
            data['kvp'],
            data['image_count'],
            data['file_path_sample']
        ))
    
    cursor.executemany('''
        INSERT INTO ct_series 
        (series_uid, patient_id, study_date, series_description, modality, kernel, slice_thickness, kvp, image_count, file_path_sample)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', rows)
    
    return total_files, dicom_files, len(series_registry)

def generate_report(cursor, total_files, dicom_files, unique_series_count):
    """Generates the Markdown report."""
    
    cursor.execute("SELECT COUNT(DISTINCT patient_id) FROM ct_series")
    unique_patients = cursor.fetchone()[0]
    
    lines = []
    lines.append("# Task 1: CT 全量数据普查报告")
    lines.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    lines.append("## 1. 概览 (Overview)")
    lines.append(f"* **总扫描文件数 (Total Files)**: {total_files}")
    lines.append(f"* **有效DICOM文件数 (Valid DICOM)**: {dicom_files}")
    lines.append(f"* **提取到的序列数 (Unique Series)**: {unique_series_count}")
    lines.append(f"* **独立病人ID数 (Unique Patient IDs)**: {unique_patients}\n")
    
    lines.append("## 2. 命名法分布 (Distribution)")
    
    # Kernel
    lines.append("### Kernel (Top 20)")
    lines.append("| Kernel | Count |")
    lines.append("| :--- | :--- |")
    cursor.execute("SELECT kernel, COUNT(*) as c FROM ct_series GROUP BY kernel ORDER BY c DESC LIMIT 20")
    for row in cursor.fetchall():
        val = row[0] if row[0] else "(Empty)"
        lines.append(f"| {val} | {row[1]} |")
    lines.append("")
    
    # Description
    lines.append("### Series Description (Top 20)")
    lines.append("| Description | Count |")
    lines.append("| :--- | :--- |")
    cursor.execute("SELECT series_description, COUNT(*) as c FROM ct_series GROUP BY series_description ORDER BY c DESC LIMIT 20")
    for row in cursor.fetchall():
        val = row[0] if row[0] else "(Empty)"
        lines.append(f"| {val} | {row[1]} |")
    lines.append("")
    
    # Slice Thickness
    lines.append("### Slice Thickness")
    lines.append("| Thickness (mm) | Count |")
    lines.append("| :--- | :--- |")
    cursor.execute("SELECT slice_thickness, COUNT(*) as c FROM ct_series GROUP BY slice_thickness ORDER BY slice_thickness ASC")
    for row in cursor.fetchall():
        val = row[0] if row[0] is not None else "N/A"
        lines.append(f"| {val} | {row[1]} |")
    lines.append("")

    # Modality
    lines.append("### Modality Check")
    lines.append("| Modality | Count |")
    lines.append("| :--- | :--- |")
    cursor.execute("SELECT modality, COUNT(*) as c FROM ct_series GROUP BY modality ORDER BY c DESC")
    for row in cursor.fetchall():
        lines.append(f"| {row[0]} | {row[1]} |")
    lines.append("")
    
    lines.append("## 3. 数据完整性 (Data Integrity)")
    cursor.execute("SELECT COUNT(*) FROM ct_series WHERE image_count < 10")
    small_count = cursor.fetchone()[0]
    lines.append(f"* **Count of series with < 10 images**: {small_count}")
    
    if small_count > 0:
        lines.append("\n**Sample of incomplete series (<10 images):**")
        lines.append("| Patient ID | Series UID | Description | Count | Path Sample |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        cursor.execute("SELECT patient_id, series_uid, series_description, image_count, file_path_sample FROM ct_series WHERE image_count < 10 LIMIT 10")
        for row in cursor.fetchall():
            uid_short = row[1][-6:] if row[1] else "N/A"
            path_short = os.path.basename(row[4]) if row[4] else "N/A"
            lines.append(f"| {row[0]} | ...{uid_short} | {row[2]} | {row[3]} | {path_short} |")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    logging.info(f"Report generated at {REPORT_PATH}")

def main():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        init_db(cursor)
        
        total, dicom, series = scan_and_populate(cursor, DATA_DIR)
        conn.commit()
        
        generate_report(cursor, total, dicom, series)
        update_progress("completed", total, dicom, series)
        
        conn.close()
        print(f"Done! Report saved to: {REPORT_PATH}")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
