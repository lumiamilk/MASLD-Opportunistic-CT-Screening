"""
Generate corrected Table 3-1 (基线特征表) with 4-column structure:
指标 | 总体人群(N=324) | 多模态融合队列(N=159) | 非融合队列(N=165) | P值（融合队列 vs 非融合队列）
"""
import pandas as pd
import numpy as np
import os
from scipy import stats

INPUT_CSV = r"D:\mWork\paper0\output\master_dataset_final.csv"
OUTPUT_DIR = r"D:\mWork\paper0\output\20260514"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load data ---
try:
    df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
except:
    df = pd.read_csv(INPUT_CSV, encoding='gb18030')

# --- Cohort split ---
mask = df['L_S_Ratio'].notna() | df['original_firstorder_Energy'].notna()
fusion_df = df[mask].copy()
non_fusion_df = df[~mask].copy()

n_total = len(df)
n_fusion = len(fusion_df)
n_non_fusion = len(non_fusion_df)

print(f"Total: {n_total}, Fusion: {n_fusion}, Non-Fusion: {n_non_fusion}")

# --- Formatting helpers ---
def format_p(p):
    if p < 0.001: return "<0.001"
    return f"{p:.3f}"

def format_cont(series):
    try:
        _, p = stats.shapiro(series.dropna())
        is_normal = p > 0.05
    except:
        is_normal = False
    if is_normal:
        return f"{series.mean():.2f} ± {series.std():.2f}"
    else:
        med = series.median()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        return f"{med:.2f} [{q1:.2f}-{q3:.2f}]"

def format_cat(series, target_val=1):
    count = (series == target_val).sum()
    total = len(series.dropna())
    perc = (count / total) * 100 if total > 0 else 0
    return f"{count} ({perc:.1f}%)"

# --- Chinese variable name mapping ---
var_cn = {
    'Age (years)': '年龄（岁）',
    'Male Sex': '男性',
    'BMI (kg/m²)': 'BMI（kg/m²）',
    'Type 2 Diabetes': '2型糖尿病',
    'Hypertension': '高血压',
    'ALT (U/L)': 'ALT（U/L）',
    'AST (U/L)': 'AST（U/L）',
    'PLT (10^9/L)': 'PLT（10⁹/L）',
    'TG (mmol/L)': 'TG（mmol/L）',
    'TC (mmol/L)': 'TC（mmol/L）',
    'HDL-C (mmol/L)': 'HDL-C（mmol/L）',
    'LDL-C (mmol/L)': 'LDL-C（mmol/L）',
    'Fasting Glucose (mmol/L)': '空腹血糖（mmol/L）',
}

# --- Column definitions: (display_label, column_name, type) ---
vars_map = [
    ('Age (years)', 'age', 'cont'),
    ('Male Sex', 'sex', 'cat_binary'),
    ('BMI (kg/m²)', 'BMI', 'cont'),
    ('Type 2 Diabetes', 'T2DM', 'cat_binary'),
    ('Hypertension', 'High_Blood_pressure', 'cat_binary'),
    ('ALT (U/L)', 'ALT_Val', 'cont'),
    ('AST (U/L)', 'AST_Val', 'cont'),
    ('PLT (10^9/L)', 'PLT_Val', 'cont'),
    ('TG (mmol/L)', 'TG_Val', 'cont'),
    ('TC (mmol/L)', 'TC_Val', 'cont'),
    ('HDL-C (mmol/L)', 'HDL-C_Val', 'cont'),
    ('LDL-C (mmol/L)', 'LDL-C_Val', 'cont'),
    ('Fasting Glucose (mmol/L)', 'GLU_Val', 'cont'),
]

rows = []

for label, col, vtype in vars_map:
    if col not in df.columns:
        print(f"WARNING: Column '{col}' not found, skipping '{label}'")
        continue

    total_data = df[col]
    fusion_data = fusion_df[col]
    non_fusion_data = non_fusion_df[col]

    if vtype == 'cont':
        v_total = format_cont(total_data)
        v_fusion = format_cont(fusion_data)
        v_non = format_cont(non_fusion_data)
        # t-test or Mann-Whitney
        try:
            _, p_norm = stats.shapiro(fusion_data.dropna())
            if p_norm > 0.05:
                _, p = stats.ttest_ind(fusion_data.dropna(), non_fusion_data.dropna())
            else:
                _, p = stats.mannwhitneyu(fusion_data.dropna(), non_fusion_data.dropna())
        except:
            p = 1.0
    else:  # cat_binary
        v_total = format_cat(total_data, 1)
        v_fusion = format_cat(fusion_data, 1)
        v_non = format_cat(non_fusion_data, 1)
        try:
            c1 = (fusion_data == 1).sum()
            c2 = (fusion_data == 0).sum()
            c3 = (non_fusion_data == 1).sum()
            c4 = (non_fusion_data == 0).sum()
            _, p, _, _ = stats.chi2_contingency([[c1, c3], [c2, c4]])
        except:
            p = 1.0

    rows.append({
        '指标': var_cn.get(label, label),
        f'总体人群(N={n_total})': v_total,
        f'多模态融合队列(N={n_fusion})': v_fusion,
        f'非融合队列(N={n_non_fusion})': v_non,
        'P值（融合队列 vs 非融合队列）': format_p(p),
    })

res_df = pd.DataFrame(rows)

# --- Save ---
output_path = os.path.join(OUTPUT_DIR, "task_17.1_table_3_1_baseline_v2.csv")
res_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\nSaved: {output_path}")
print(res_df.to_string(index=False))
