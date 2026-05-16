import pandas as pd
import numpy as np
import os
import json
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
JSON_PATH = os.path.join(BASE_DIR, "data", "original_data.json")
CSV_PATH = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
REPORT_MD = os.path.join(BASE_DIR, "output", "task_14_run_report.md")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_text_labels():
    logging.info("Mining human reports from JSON...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    records = []
    for entry in data:
        pid = str(entry.get('patient_id', '')).strip()
        if not pid: continue
        
        ct_text = str(entry.get('CT', '')).lower()
        us_text = str(entry.get('Bchao', '')).lower()
        
        # Rule-based extraction
        # Steatosis (Any): "脂肪肝" or "脂肪变"
        # Steatosis (Severe): ("重度" or "中重度") AND "脂肪"
        # Cirrhosis: "肝硬化"
        
        is_sev_fat = lambda t: ('重度' in t or '中重度' in t) and ('脂肪' in t)
        is_cirr = lambda t: '肝硬化' in t
        
        records.append({
            'patient_id': pid,
            'Human_CT_S_Sev': int(is_sev_fat(ct_text)),
            'Human_CT_Cirr': int(is_cirr(ct_text)),
            'Human_US_S_Sev': int(is_sev_fat(us_text)),
            'Human_US_Cirr': int(is_cirr(us_text))
        })
        
    return pd.DataFrame(records)

def train_ai_model(df, target_col, target_name):
    # Features (Fusion)
    feats = [c for c in df.columns if c.startswith('original_') or c.endswith('_R') or c.startswith('Index_')]
    feats += ['age', 'BMI', 'T2DM', 'L_S_Ratio', 'Muscle_Mean_HU', 'VAT_SAT_Ratio']
    feats = [c for c in feats if c in df.columns]
    
    X = df[feats].values
    y = df[target_col].astype(int).values
    ids = df['patient_id'].values
    
    # Store predictions
    # Initialize with -1
    y_pred_all = np.full(len(y), -1)
    
    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        
        # Preprocessing
        imputer = KNNImputer(n_neighbors=5)
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # SMOTE
        k = min(np.sum(y_train) - 1, 5)
        if k > 0:
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train)
        
        # Predict
        preds = clf.predict(X_test)
        y_pred_all[test_idx] = preds
        
    return pd.DataFrame({
        'patient_id': ids,
        f'AI_{target_name}_Pred': y_pred_all
    })

def calc_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return acc, sens, spec

def main():
    if not os.path.exists(CSV_PATH): return
    
    # 1. Load Data
    try: df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    except: df = pd.read_csv(CSV_PATH, encoding='gb18030')
    df['patient_id'] = df['patient_id'].astype(str).str.strip()
    
    # Subset (Fusion)
    has_valid_ct = df['L_S_Ratio'].notna() | df['original_firstorder_Energy'].notna()
    df_fusion = df[has_valid_ct].copy()
    logging.info(f"Fusion Subset: {len(df_fusion)}")
    
    # 2. Get Human Labels
    df_human = load_text_labels()
    df_merged = pd.merge(df_fusion, df_human, on='patient_id', how='left')
    
    # 3. Train AI Models & Get Predictions
    # Target 1: S_Sev
    df_merged['Gold_S_Sev'] = (df_merged['Steatosis_Grade'] == 3).astype(int)
    ai_s = train_ai_model(df_merged, 'Gold_S_Sev', 'S_Sev')
    df_final = pd.merge(df_merged, ai_s, on='patient_id')
    
    # Target 2: F_Cirr
    df_final['Gold_F_Cirr'] = (df_final['Fibrosis_Stage'] == 4).astype(int)
    ai_f = train_ai_model(df_final, 'Gold_F_Cirr', 'F_Cirr')
    df_final = pd.merge(df_final, ai_f, on='patient_id')
    
    # 4. Compare & Report
    lines = []
    lines.append("# Task 14: Human vs AI Performance Report")
    lines.append(f"Generated on: {pd.Timestamp.now()}")
    lines.append(f"Analysis Sample Size: {len(df_final)}")
    lines.append("")
    
    # Define tasks
    tasks = [
        {'name': 'Severe Steatosis (S=3)', 'gold': 'Gold_S_Sev', 
         'human_ct': 'Human_CT_S_Sev', 'human_us': 'Human_US_S_Sev', 'ai': 'AI_S_Sev_Pred'},
        {'name': 'Cirrhosis (F=4)', 'gold': 'Gold_F_Cirr', 
         'human_ct': 'Human_CT_Cirr', 'human_us': 'Human_US_Cirr', 'ai': 'AI_F_Cirr_Pred'}
    ]
    
    for t in tasks:
        lines.append(f"## Target: {t['name']}")
        lines.append("| Diagnostic Method | Sensitivity (Recall) | Specificity | Accuracy |")
        lines.append("| :--- | :--- | :--- | :--- |")
        
        # Calculate for Human CT
        valid_h_ct = df_final[df_final[t['human_ct']].notna()]
        acc, sens, spec = calc_metrics(valid_h_ct[t['gold']], valid_h_ct[t['human_ct']])
        lines.append(f"| Human CT Report | {sens:.3f} | {spec:.3f} | {acc:.3f} |")
        
        # Calculate for Human US
        valid_h_us = df_final[df_final[t['human_us']].notna()]
        acc, sens, spec = calc_metrics(valid_h_us[t['gold']], valid_h_us[t['human_us']])
        lines.append(f"| Human US Report | {sens:.3f} | {spec:.3f} | {acc:.3f} |")
        
        # Calculate for AI
        acc, sens, spec = calc_metrics(df_final[t['gold']], df_final[t['ai']])
        lines.append(f"| **AI Model (Fusion)** | **{sens:.3f}** | **{spec:.3f}** | **{acc:.3f}** |")
        
        lines.append("")
        
        # Case Finding: AI Correct, Human CT Wrong (False Negative)
        # Condition: Gold=1, Human=0, AI=1
        mask_win = (df_final[t['gold']] == 1) & (df_final[t['human_ct']] == 0) & (df_final[t['ai']] == 1)
        winners = df_final[mask_win]['patient_id'].tolist()
        
        if winners:
            lines.append(f"### AI Super-Human Cases (Human Missed, AI Found): {len(winners)}")
            lines.append(f"Patient IDs: `{', '.join(winners)}`")
            lines.append("*These cases represent the core clinical value: detecting silent disease.*")
        else:
            lines.append("No super-human cases found for this target.")
        lines.append("")

    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    logging.info(f"Report saved: {REPORT_MD}")

if __name__ == "__main__":
    main()
