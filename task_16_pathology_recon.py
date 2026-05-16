import os
import pandas as pd
import logging
import re
from glob import glob

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
PATHOLOGY_DIR = os.path.join(BASE_DIR, "data", "临床liver 切片")
MAPPING_XLSX = os.path.join(BASE_DIR, "data", "肝穿[2017.1.1-2025.12.31].xlsx")
MASTER_CSV = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
OUTPUT_INVENTORY = os.path.join(BASE_DIR, "output", "task_16_pathology_inventory.csv")
OUTPUT_REPORT = os.path.join(BASE_DIR, "output", "task_16_recon_report.md")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_master_ids():
    if not os.path.exists(MASTER_CSV):
        logging.error("Master dataset not found.")
        return set()
    try:
        df = pd.read_csv(MASTER_CSV)
        # Ensure patient_id is string
        return set(df['patient_id'].astype(str).str.strip().tolist())
    except Exception as e:
        logging.error(f"Error reading master csv: {e}")
        return set()

def load_mapping():
    if not os.path.exists(MAPPING_XLSX):
        logging.error(f"Mapping file not found: {MAPPING_XLSX}")
        return {}
    
    logging.info(f"Loading mapping from {MAPPING_XLSX}...")
    try:
        # Load all sheets? Usually first sheet is enough.
        df = pd.read_excel(MAPPING_XLSX)
        
        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()
        
        # Look for target columns
        path_col = next((c for c in df.columns if '病理号' in c), None)
        pid_col = next((c for c in df.columns if '住院号' in c), None)
        
        if not path_col or not pid_col:
            logging.error(f"Columns not found. Available: {df.columns}")
            return {}
            
        # Build dict
        mapping = {}
        for _, row in df.iterrows():
            path_id = str(row[path_col]).strip()
            pid = str(row[pid_col]).strip()
            
            # Basic validation
            if path_id and pid and path_id.lower() != 'nan' and pid.lower() != 'nan':
                mapping[path_id] = pid
                
        logging.info(f"Loaded {len(mapping)} mappings.")
        return mapping
    except Exception as e:
        logging.error(f"Error loading excel: {e}")
        return {}

def scan_pathology_files():
    logging.info(f"Scanning directory: {PATHOLOGY_DIR}")
    
    records = []
    
    # Extensions for WSI (Whole Slide Image) or standard images
    valid_exts = {'.svs', '.kfb', '.tif', '.tiff', '.ndpi', '.mrxs', '.jpg', '.jpeg', '.png', '.bmp'}
    
    for root, dirs, files in os.walk(PATHOLOGY_DIR):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_exts:
                full_path = os.path.join(root, file)
                
                # Extract Pathology ID logic
                # Pattern: Looks for L + digits (L220187) or just digits if folder has context
                # Often in folder name: "I级脂肪肝 L220187"
                # Or filename: "L220187.kfb"
                
                # Try filename first
                match = re.search(r'([A-Za-z]?\d{5,})', file)
                path_id = match.group(1) if match else None
                
                # If not in filename, try folder name
                if not path_id:
                    folder_name = os.path.basename(root)
                    match_folder = re.search(r'([A-Za-z]?\d{5,})', folder_name)
                    path_id = match_folder.group(1) if match_folder else None
                
                records.append({
                    'Path': full_path,
                    'Filename': file,
                    'Extension': ext,
                    'Size_MB': os.path.getsize(full_path) / (1024*1024),
                    'Extracted_PathID': path_id
                })
                
    logging.info(f"Found {len(records)} pathology image files.")
    return pd.DataFrame(records)

def main():
    # 1. Load Resources
    master_ids = load_master_ids()
    path_map = load_mapping()
    
    # 2. Scan Files
    df_files = scan_pathology_files()
    
    if df_files.empty:
        logging.warning("No pathology files found.")
        return

    # 3. Match
    df_files['Patient_ID'] = df_files['Extracted_PathID'].map(path_map)
    
    # 4. Status
    def get_status(row):
        if pd.isna(row['Patient_ID']):
            return "No_Patient_ID_Found"
        if row['Patient_ID'] in master_ids:
            return "Match_In_Master"
        return "Patient_ID_Found_But_No_CT"
        
    df_files['Match_Status'] = df_files.apply(get_status, axis=1)
    
    # 5. Output
    df_files.to_csv(OUTPUT_INVENTORY, index=False)
    logging.info(f"Inventory saved to {OUTPUT_INVENTORY}")
    
    # 6. Generate Report
    generate_report(df_files)

def generate_report(df):
    stats = df['Match_Status'].value_counts()
    ext_stats = df['Extension'].value_counts()
    
    match_count = stats.get('Match_In_Master', 0)
    
    lines = []
    lines.append("# Task 16.1: Pathology Data Reconnaissance")
    lines.append(f"Generated on: {pd.Timestamp.now()}")
    lines.append("")
    
    lines.append("## 1. File Inventory")
    lines.append(f"*   **Total Images Found**: {len(df)}")
    lines.append(f"*   **Total Size**: {df['Size_MB'].sum() / 1024:.2f} GB")
    lines.append("")
    lines.append("### Format Distribution")
    lines.append(ext_stats.to_markdown())
    lines.append("")
    
    lines.append("## 2. Matching Status")
    lines.append(f"Matching logic: Filename/Folder -> Pathology ID -> Excel Map -> Patient ID -> Master Dataset")
    lines.append("")
    lines.append(stats.to_markdown())
    lines.append("")
    
    lines.append("## 3. The 'Wet Lab' Cohort")
    lines.append(f"**We found {match_count} patients who have:**")
    lines.append("1.  Clinical Data")
    lines.append("2.  CT Radiomics")
    lines.append("3.  Body Composition")
    lines.append("4.  **Pathology Slides (Microscope Images)**")
    lines.append("")
    
    if match_count > 0:
        sample = df[df['Match_Status'] == 'Match_In_Master'].head(5)
        lines.append("### Sample Matched Files")
        lines.append(sample[['Filename', 'Extracted_PathID', 'Patient_ID', 'Size_MB']].to_markdown(index=False))
        
    lines.append("")
    lines.append("## 4. Next Steps")
    lines.append("*   If we have `.kfb` or `.svs` files, we need `openslide` or specific drivers to read them.")
    lines.append("*   We can extract patches from these slides to show 'Ground Truth' visualization.")
    
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    logging.info(f"Report saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
