import pandas as pd
import numpy as np
import os
import logging

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
INPUT_CSV = os.path.join(BASE_DIR, "output", "master_dataset.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "output", "master_dataset_with_indices.csv")
REPORT_MD = os.path.join(BASE_DIR, "output", "task_6_run_report.md")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input file not found: {INPUT_CSV}")
        return

    try:
        df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_CSV, encoding='gb18030')
        
    logging.info(f"Loaded {len(df)} records from {INPUT_CSV}")

    # Helper to safe-divide
    def safe_div(a, b):
        return np.where(b != 0, a / b, np.nan)

    # 1. FIB-4
    # Formula: (Age * AST) / (PLT * sqrt(ALT))
    if all(c in df.columns for c in ['age', 'AST_Val', 'PLT_Val', 'ALT_Val']):
        df['Index_FIB4'] = (df['age'] * df['AST_Val']) / (df['PLT_Val'] * np.sqrt(df['ALT_Val']))
    else:
        logging.warning("Missing columns for FIB-4 calculation.")

    # 2. APRI
    # Formula: ((AST / 40) * 100) / PLT
    if all(c in df.columns for c in ['AST_Val', 'PLT_Val']):
        df['Index_APRI'] = ((df['AST_Val'] / 40) * 100) / df['PLT_Val']

    # 3. TyG
    # Formula: ln( (TG * 88.5 * GLU * 18) / 2 )
    if all(c in df.columns for c in ['TG_Val', 'GLU_Val']):
        # Using log in numpy (natural log)
        tg_mg = df['TG_Val'] * 88.5
        glu_mg = df['GLU_Val'] * 18
        df['Index_TyG'] = np.log((tg_mg * glu_mg) / 2)
    
    # 4. TyG-BMI
    if 'Index_TyG' in df.columns and 'BMI' in df.columns:
        df['Index_TyG_BMI'] = df['Index_TyG'] * df['BMI']

    # 5. NLR (Neutrophil / Lymphocyte Ratio)
    if all(c in df.columns for c in ['NE#_Val', 'LY#_Val']):
        df['Index_NLR'] = safe_div(df['NE#_Val'], df['LY#_Val'])

    # 6. PLR (Platelet / Lymphocyte Ratio)
    if all(c in df.columns for c in ['PLT_Val', 'LY#_Val']):
        df['Index_PLR'] = safe_div(df['PLT_Val'], df['LY#_Val'])

    # 7. SII (Systemic Immune-Inflammation Index)
    # Formula: (PLT * NE#) / LY#
    if all(c in df.columns for c in ['PLT_Val', 'NE#_Val', 'LY#_Val']):
        df['Index_SII'] = safe_div(df['PLT_Val'] * df['NE#_Val'], df['LY#_Val'])

    # 8. AST/ALT Ratio
    if all(c in df.columns for c in ['AST_Val', 'ALT_Val']):
        df['Index_AST_ALT'] = safe_div(df['AST_Val'], df['ALT_Val'])

    # Save data
    df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Saved master dataset with indices to {OUTPUT_CSV}")

    # Generate Report
    generate_report(df)

def generate_report(df):
    index_cols = [c for c in df.columns if c.startswith('Index_')]
    
    stats = []
    for col in index_cols:
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        stats.append({
            'Index': col,
            'Count': len(series),
            'Coverage (%)': (len(series) / len(df)) * 100,
            'Mean': series.mean(),
            'Min': series.min(),
            'Max': series.max()
        })
    
    stats_df = pd.DataFrame(stats)
    
    report_content = f"""# Task 6 Run Report: Clinical Index Calculation

Generated on: {pd.Timestamp.now()}

## 1. Index Coverage and Stats
| Index | Count | Coverage (%) | Mean | Min | Max |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, row in stats_df.iterrows():
        report_content += f"| {row['Index']} | {int(row['Count'])} | {row['Coverage (%)']:.2f}% | {row['Mean']:.4f} | {row['Min']:.4f} | {row['Max']:.4f} |\n"

    report_content += f"""
## 2. Summary
*   **Input**: {len(df)} patients.
*   **Output**: `{OUTPUT_CSV}`
*   **Key Notes**:
    *   FIB-4 and APRI are indicators for liver fibrosis.
    *   TyG and TyG-BMI are markers for insulin resistance and MAFLD.
    *   NLR, PLR, SII are systemic inflammation markers.
"""
    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write(report_content)
    logging.info(f"Report saved to {REPORT_MD}")

if __name__ == "__main__":
    main()
