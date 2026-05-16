import pandas as pd
import os
import logging

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
MASTER_CSV = os.path.join(BASE_DIR, "output", "master_dataset.csv")
REPORT_MD = os.path.join(BASE_DIR, "output", "task_5_missing_data_report.md")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def df_to_md(df):
    """Simple manual conversion of DataFrame to Markdown table."""
    cols = df.columns.tolist()
    header = "| " + " | ".join(map(str, cols)) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(map(lambda x: f"{x:.2f}" if isinstance(x, float) else str(x), row.values)) + " |")
    return "\n".join([header, sep] + rows)

def analyze_missingness():
    logging.info(f"Loading master dataset from {MASTER_CSV}...")
    df = pd.read_csv(MASTER_CSV)
    total_rows = len(df)
    
    # Calculate missingness
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / total_rows) * 100
    
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing_count.values,
        'Missing_Percentage': missing_pct.values
    })
    
    # Categorize columns
    def categorize(col):
        if col in ['patient_id', 'age', 'sex', 'BMI', 'T2DM', 'High_Blood_pressure']:
            return 'Basic Info'
        if col in ['Fibrosis_Stage', 'Inflammation_Grade', 'Steatosis_Grade', 'NAS', 'SAF']:
            return 'Pathology'
        if col.startswith('US_'):
            return 'Ultrasound'
        if col.endswith('_R'):
            return 'Lab Ratios'
        if col.endswith('_Val'):
            return 'Lab Values'
        if col.startswith('original_'):
            return 'Radiomics (CT)'
        if col.startswith('diagnostics_'):
            return 'Radiomics Metadata'
        return 'Other'

    missing_df['Category'] = missing_df['Column'].apply(categorize)
    
    # Generate Report
    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write("# Master Dataset 缺失值分析报告\n\n")
        f.write(f"**分析样本总数**: {total_rows}\n")
        f.write(f"**生成时间**: {pd.Timestamp.now()}\n\n")
        
        # Summary table by category
        f.write("## 1. 各维度特征缺失概况\n")
        summary = missing_df.groupby('Category').agg({
            'Column': 'count',
            'Missing_Percentage': 'mean'
        }).reset_index().rename(columns={'Column': 'Count', 'Missing_Percentage': 'Avg_Missing_Pct'})
        f.write(df_to_md(summary) + "\n\n")
        
        # Specific High-Value Columns
        f.write("## 2. 核心指标缺失详情\n")
        core_cols = [
            'age', 'sex', 'BMI', 'T2DM', 'High_Blood_pressure',
            'Fibrosis_Stage', 'Inflammation_Grade', 'Steatosis_Grade', 'NAS', 'SAF',
            'US_Echo', 'US_Atten', 'US_Spleen', 'US_Liver_Size'
        ]
        core_df = missing_df[missing_df['Column'].isin(core_cols)].sort_values('Missing_Percentage')
        f.write(df_to_md(core_df[['Column', 'Missing_Count', 'Missing_Percentage']]) + "\n\n")
        
        # Radiomics Coverage
        f.write("## 3. 影像组学 (CT) 覆盖分析\n")
        rad_sample = 'original_firstorder_Energy'
        if rad_sample in df.columns:
            rad_missing = missing_df[missing_df['Column'] == rad_sample]['Missing_Percentage'].values[0]
            f.write(f"*   **CT 特征覆盖率**: {100 - rad_missing:.2f}%\n")
            f.write(f"*   **匹配成功的患者数**: {total_rows - int(missing_count[rad_sample])}\n\n")
        
        # Top Missing Labs
        f.write("## 4. 缺失率最高的前 20 个实验室指标 (Ratio)\n")
        lab_r_df = missing_df[missing_df['Category'] == 'Lab Ratios'].sort_values('Missing_Percentage', ascending=False)
        f.write(df_to_md(lab_r_df[['Column', 'Missing_Count', 'Missing_Percentage']].head(20)) + "\n\n")
        
        f.write("---\n*报告结束*")

    logging.info(f"Report saved to {REPORT_MD}")

if __name__ == "__main__":
    analyze_missingness()
