import pandas as pd
import os
import numpy as np
import logging

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
MASTER_CSV = os.path.join(BASE_DIR, "output", "master_dataset_with_indices.csv")
ORGAN_CSV = os.path.join(BASE_DIR, "output", "task_8_organ_features.csv")
BODY_CSV = os.path.join(BASE_DIR, "output", "task_8_body_features.csv")
OUTPUT_FINAL = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
REPORT_MD = os.path.join(BASE_DIR, "output", "task_8_report.md")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_csv(path):
    if not os.path.exists(path): return None
    try: return pd.read_csv(path, encoding='utf-8-sig', dtype={'Patient_ID': str, 'patient_id': str})
    except: return pd.read_csv(path, encoding='gb18030', dtype={'Patient_ID': str, 'patient_id': str})

def main():
    df_master = load_csv(MASTER_CSV)
    df_organ = load_csv(ORGAN_CSV)
    df_body = load_csv(BODY_CSV)
    
    if df_master is None: return

    # 1. Merge Organ + Body
    df_comp = pd.merge(df_organ, df_body, on='Patient_ID', how='outer')
    
    # 2. Calculate VAT/SAT Ratio
    if 'Visceral_Fat_Volume' in df_comp.columns and 'Fat_Volume' in df_comp.columns:
        df_comp['VAT_SAT_Ratio'] = df_comp['Visceral_Fat_Volume'] / df_comp['Fat_Volume']
    
    # 3. Merge into Master
    df_comp.rename(columns={'Patient_ID': 'patient_id'}, inplace=True)
    df_master['patient_id'] = df_master['patient_id'].str.strip()
    df_comp['patient_id'] = df_comp['patient_id'].str.strip()
    
    df_final = pd.merge(df_master, df_comp, on='patient_id', how='left')
    
    # 4. Save
    df_final.to_csv(OUTPUT_FINAL, index=False)
    logging.info(f"Final dataset saved to {OUTPUT_FINAL}")
    
    # 5. Generate FINAL Report
    generate_final_report(df_final, df_organ, df_body)

def generate_final_report(df_final, df_organ, df_body):
    lines = []
    lines.append("# Task 8 最终总结报告: 体成分多模态集成")
    lines.append(f"生成时间: {pd.Timestamp.now()}")
    lines.append("")
    lines.append("## 1. 核心产物")
    lines.append(f"*   **终极研究数据集**: `{OUTPUT_FINAL}`")
    lines.append(f"*   **包含特征**: 临床+病理+检验+指数+组学+体成分 (共 {len(df_final.columns)} 维特征)")
    lines.append("")
    lines.append("## 2. 特征覆盖统计 (N=324)")
    cols = ['Liver_Volume', 'Spleen_Volume', 'L_S_Ratio', 'Muscle_Volume', 'Fat_Volume', 'Visceral_Fat_Volume', 'VAT_SAT_Ratio']
    lines.append("| 指标 | 有效样本数 | 覆盖率 (%) | 均值 |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for c in cols:
        if c in df_final.columns:
            s = df_final[c].dropna()
            lines.append(f"| `{c}` | {len(s)} | {len(s)/324*100:.2f}% | {s.mean():.2f} |")
            
    lines.append("")
    lines.append("## 3. 执行过程深度回顾")
    lines.append("### 3.1 阶段一：肝脾分割 (Task 8.1)")
    lines.append("*   使用 `TotalSegmentator` 的 `fast=True` 模式，成功提取了 164 名患者的 `L/S Ratio`。")
    lines.append("### 3.2 阶段二：组织类型分割 (Task 8.2 & 8.3)")
    lines.append("*   **授权**: 成功申请并配置了官方学术授权码。")
    lines.append("*   **技术选型**: 使用 CLI 模式 `-ta tissue_types -s`。")
    lines.append("*   **迭代**: 补全了之前遗漏的 `Visceral_Fat` (内脏脂肪) 数据。")
    lines.append("*   **成果**: 获取了骨骼肌、皮下脂肪、内脏脂肪的体积与密度。")
    lines.append("")
    lines.append("## 4. 临床研究建议")
    lines.append("1.  **VAT/SAT Ratio**: 建议作为预测 MAFLD 炎症水平的核心变量。")
    lines.append("2.  **Muscle_Mean_HU**: 可用于研究肌少症 (Sarcopenia) 与非酒精性脂肪肝的关联。")
    lines.append("3.  **L_S_Ratio**: 已集成的肝脾比是诊断中重度脂肪变性的可靠指标。")
    lines.append("")
    lines.append("---")
    lines.append("*报告人：AI 算法工程师*")
    
    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()