import pandas as pd
import json
import os
import re

# Paths
BASE_DIR = "/mnt/d/mWork/paper0"
JSON_PATH = os.path.join(BASE_DIR, "data", "original_data.json")
CSV_PATH = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
REPORT_OUTPUT = os.path.join(BASE_DIR, "output", "task_0_audit_report.md")

def extract_doctor_grade(text):
    if not isinstance(text, str):
        return None, None
    
    text = text.replace(" ", "") # Remove spaces for easier matching
    
    # 1. Steatosis Grade
    # Priority: Severe > Moderate > Mild/Fatty Liver > None
    steatosis_grade = 0
    if "重度" in text:
        steatosis_grade = 3
    elif "中度" in text:
        steatosis_grade = 2
    elif "脂肪肝" in text:
        steatosis_grade = 1
    
    # 2. Cirrhosis (Fibrosis Stage 4 equivalent)
    cirrhosis = 0
    if "肝硬化" in text:
        cirrhosis = 1
        
    return steatosis_grade, cirrhosis

def main():
    print(f"Loading JSON from {JSON_PATH}...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loading CSV from {CSV_PATH}...")
    df_master = pd.read_csv(CSV_PATH, dtype={'patient_id': str})
    df_master['patient_id'] = df_master['patient_id'].str.strip()
    
    # Parse JSON
    records = []
    for item in data:
        pid = item.get('patient_id')
        if not pid: continue
        
        # Bchao
        b_text = item.get('Bchao', '')
        if isinstance(b_text, str) and len(b_text) > 10:
             b_s, b_c = extract_doctor_grade(b_text)
        else:
             b_s, b_c = None, None
        
        # CT
        ct_text = item.get('CT', '')
        if isinstance(ct_text, str) and len(ct_text) > 10:
             ct_s, ct_c = extract_doctor_grade(ct_text)
        else:
             ct_s, ct_c = None, None
        
        records.append({
            'patient_id': str(pid).strip(),
            'Doctor_US_Steatosis': b_s,
            'Doctor_US_Cirrhosis': b_c,
            'Doctor_CT_Steatosis': ct_s,
            'Doctor_CT_Cirrhosis': ct_c,
            'Has_US_Report': bool(b_s is not None),
            'Has_CT_Report': bool(ct_s is not None)
        })
        
    df_doctors = pd.DataFrame(records)
    
    # Merge
    print("Merging data...")
    merged = pd.merge(df_master, df_doctors, on='patient_id', how='inner')
    
    # Analyze Steatosis (Gold Standard: Steatosis_Grade)
    valid_s = merged.dropna(subset=['Steatosis_Grade'])
    try:
        valid_s['Steatosis_Grade'] = valid_s['Steatosis_Grade'].astype(float).astype(int)
    except:
        pass # In case of weird strings
    
    # Analyze Fibrosis (Gold Standard: Fibrosis_Stage)
    valid_f = merged.dropna(subset=['Fibrosis_Stage'])
    try:
        valid_f['Fibrosis_Stage'] = valid_f['Fibrosis_Stage'].astype(float).astype(int)
    except:
        pass

    with open(REPORT_OUTPUT, 'w', encoding='utf-8') as f:
        f.write("# Task 0: Radiological Report Audit\n\n")
        f.write(f"Total Patients Merged: {len(merged)}\n\n")
        
        # --- Steatosis Analysis ---
        f.write("## 1. Steatosis Diagnosis Performance\n")
        
        def calc_miss_rate(df, gold_col, gold_val, doc_col, cutoff):
            target_group = df[df[gold_col] == gold_val]
            if len(target_group) == 0: return 0, 0
            
            target_group_with_report = target_group.dropna(subset=[doc_col])
            n_total = len(target_group_with_report)
            if n_total == 0: return 0, 0
            
            missed = target_group_with_report[target_group_with_report[doc_col] < cutoff]
            n_missed = len(missed)
            return n_missed, n_total

        # US Steatosis
        f.write("### Ultrasound (B-Mode)\n")
        s3_miss, s3_tot = calc_miss_rate(valid_s, 'Steatosis_Grade', 3, 'Doctor_US_Steatosis', 3)
        s2_miss, s2_tot = calc_miss_rate(valid_s, 'Steatosis_Grade', 2, 'Doctor_US_Steatosis', 2)
        
        pct_s3 = (s3_miss/s3_tot*100) if s3_tot > 0 else 0
        pct_s2 = (s2_miss/s2_tot*100) if s2_tot > 0 else 0

        f.write(f"- **Severe (S3) Miss Rate**: {s3_miss}/{s3_tot} ({pct_s3:.1f}%)"
                f" (Doctor called it < Severe)\n")
        f.write(f"- **Moderate (S2) Miss Rate**: {s2_miss}/{s2_tot} ({pct_s2:.1f}%)"
                f" (Doctor called it < Moderate)\n")
        
        ctab_us = pd.crosstab(valid_s['Steatosis_Grade'], valid_s['Doctor_US_Steatosis'].fillna(-1))
        f.write("\n**Confusion Matrix (Pathology vs Ultrasound)**:\n")
        f.write("(-1 means No Report/Parse Fail)\n")
        f.write(ctab_us.to_markdown() + "\n\n")
        
        # CT Steatosis
        f.write("### CT Scan (Plain)\n")
        s3_miss_ct, s3_tot_ct = calc_miss_rate(valid_s, 'Steatosis_Grade', 3, 'Doctor_CT_Steatosis', 3)
        s2_miss_ct, s2_tot_ct = calc_miss_rate(valid_s, 'Steatosis_Grade', 2, 'Doctor_CT_Steatosis', 2)
        
        pct_s3_ct = (s3_miss_ct/s3_tot_ct*100) if s3_tot_ct > 0 else 0
        pct_s2_ct = (s2_miss_ct/s2_tot_ct*100) if s2_tot_ct > 0 else 0

        f.write(f"- **Severe (S3) Miss Rate**: {s3_miss_ct}/{s3_tot_ct} ({pct_s3_ct:.1f}%)"
                f" (Doctor called it < Severe)\n")
        f.write(f"- **Moderate (S2) Miss Rate**: {s2_miss_ct}/{s2_tot_ct} ({pct_s2_ct:.1f}%)"
                f" (Doctor called it < Moderate)\n")
        
        ctab_ct = pd.crosstab(valid_s['Steatosis_Grade'], valid_s['Doctor_CT_Steatosis'].fillna(-1))
        f.write("\n**Confusion Matrix (Pathology vs CT)**:\n")
        f.write(ctab_ct.to_markdown() + "\n\n")

        # --- Fibrosis Analysis ---
        f.write("## 2. Cirrhosis/Fibrosis Diagnosis Performance\n")
        
        f4_group = valid_f[valid_f['Fibrosis_Stage'] == 4]
        
        # US
        f4_us_report = f4_group.dropna(subset=['Doctor_US_Cirrhosis'])
        if len(f4_us_report) > 0:
            missed_us = f4_us_report[f4_us_report['Doctor_US_Cirrhosis'] == 0]
            f.write(f"- **Cirrhosis (F4) Miss Rate (US)**: {len(missed_us)}/{len(f4_us_report)}"
                    f" ({len(missed_us)/len(f4_us_report)*100:.1f}%)\n")
        else:
             f.write("- **Cirrhosis (F4) Miss Rate (US)**: N/A\n")
             
        # CT
        f4_ct_report = f4_group.dropna(subset=['Doctor_CT_Cirrhosis'])
        if len(f4_ct_report) > 0:
            missed_ct = f4_ct_report[f4_ct_report['Doctor_CT_Cirrhosis'] == 0]
            f.write(f"- **Cirrhosis (F4) Miss Rate (CT)**: {len(missed_ct)}/{len(f4_ct_report)}"
                    f" ({len(missed_ct)/len(f4_ct_report)*100:.1f}%)\n")
        else:
             f.write("- **Cirrhosis (F4) Miss Rate (CT)**: N/A\n")

    print(f"Report generated at {REPORT_OUTPUT}")

if __name__ == "__main__":
    main()
