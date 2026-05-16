import pandas as pd
import numpy as np
import os
import json
import logging
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
MASTER_CSV = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
ORIGINAL_JSON = os.path.join(BASE_DIR, "data", "original_data.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "thesis_assets", "tables")
MODEL_LEADERBOARD = os.path.join(BASE_DIR, "output", "model_arena_leaderboard.csv")
QUANT_RESULTS = os.path.join(BASE_DIR, "output", "task_16_quant_results_v5.3.csv")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data():
    try: df = pd.read_csv(MASTER_CSV, encoding='utf-8-sig')
    except: df = pd.read_csv(MASTER_CSV, encoding='gb18030')
    return df

def get_fusion_cohort(df):
    # Logic from task_12: Valid CT data
    # 'L_S_Ratio' is a derived CT feature, 'original_firstorder_Energy' is a radiomics feature
    # Using L_S_Ratio is a good proxy for "Body Composition Analysis Done"
    # Using radiomics feature is a proxy for "Radiomics Done"
    mask = df['L_S_Ratio'].notna() | df['original_firstorder_Energy'].notna()
    return df[mask].copy()

def format_p_value(p):
    if p < 0.001: return "<0.001"
    else: return f"{p:.3f}"

def format_cont_var(series):
    # Check normality
    try:
        _, p = stats.shapiro(series.dropna())
        is_normal = p > 0.05
    except:
        is_normal = False # Fallback for large N or errors
    
    if is_normal:
        m = series.mean()
        s = series.std()
        return f"{m:.2f} ± {s:.2f}"
    else:
        med = series.median()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        return f"{med:.2f} [{q1:.2f}-{q3:.2f}]"

def format_cat_var(series, target_val=1):
    count = (series == target_val).sum()
    total = len(series.dropna())
    perc = (count / total) * 100 if total > 0 else 0
    return f"{count} ({perc:.1f}%)"

# --- Table Generators ---

def generate_table_3_1(df):
    logging.info("Generating Table 3-1: Baseline Characteristics")
    
    fusion_df = get_fusion_cohort(df)
    non_fusion_df = df[~df.index.isin(fusion_df.index)]
    
    n_total = len(df)
    n_fusion = len(fusion_df)
    n_non_fusion = len(non_fusion_df)
    
    # Variables mapping
    # 'Variable Name': ('Column Name', 'Type') -> Type: 'cont' or 'cat'
    vars_map = {
        'Age (years)': ('age', 'cont'),
        'Male Sex': ('sex', 'cat_binary'),
        'BMI (kg/m²)': ('BMI', 'cont'),
        'Type 2 Diabetes': ('T2DM', 'cat_binary'),
        'Hypertension': ('High_Blood_pressure', 'cat_binary'),
        'ALT (U/L)': ('ALT_Val', 'cont'),
        'AST (U/L)': ('AST_Val', 'cont'),
        'PLT (10^9/L)': ('PLT_Val', 'cont'),
        'TG (mmol/L)': ('TG_Val', 'cont'),
        'TC (mmol/L)': ('TC_Val', 'cont'),
        'HDL-C (mmol/L)': ('HDL-C_Val', 'cont'),
        'LDL-C (mmol/L)': ('LDL-C_Val', 'cont'),
        'Fasting Glucose (mmol/L)': ('GLU_Val', 'cont')
    }
    
    rows = []
    
    for label, (col, vtype) in vars_map.items():
        if col not in df.columns:
            logging.warning(f"Column {col} not found for {label}")
            continue
            
        # Total Cohort
        total_data = df[col]
        if vtype == 'cont':
            val_total = format_cont_var(total_data)
        elif vtype == 'cat_binary':
            val_total = format_cat_var(total_data, 1)
            
        # Fusion Cohort
        fusion_data = fusion_df[col]
        if vtype == 'cont':
            val_fusion = format_cont_var(fusion_data)
        elif vtype == 'cat_binary':
            val_fusion = format_cat_var(fusion_data, 1)
            
        # Non-Fusion Cohort
        non_fusion_data = non_fusion_df[col]
        if vtype == 'cont':
            val_non_fusion = format_cont_var(non_fusion_data)
        elif vtype == 'cat_binary':
            val_non_fusion = format_cat_var(non_fusion_data, 1)
        
        # P-value: Fusion vs Non-Fusion
        if vtype == 'cont':
            try:
                _, p_norm = stats.shapiro(fusion_data.dropna())
                if p_norm > 0.05:
                    _, p = stats.ttest_ind(fusion_data.dropna(), non_fusion_data.dropna())
                else:
                    _, p = stats.mannwhitneyu(fusion_data.dropna(), non_fusion_data.dropna())
            except:
                p = 1.0
        elif vtype == 'cat_binary':
            try:
                c1 = (fusion_data == 1).sum()
                c2 = (fusion_data == 0).sum()
                c3 = (non_fusion_data == 1).sum()
                c4 = (non_fusion_data == 0).sum()
                _, p, _, _ = stats.chi2_contingency([[c1, c3], [c2, c4]])
            except:
                p = 1.0
        
        rows.append({
            'Variable': label,
            f'Total (N={n_total})': val_total,
            f'Fusion Cohort (N={n_fusion})': val_fusion,
            f'Non-Fusion Cohort (N={n_non_fusion})': val_non_fusion,
            'P-value (Fusion vs Non-Fusion)': format_p_value(p)
        })
        
    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUTPUT_DIR, "task_17.1_table_3_1_baseline.csv"), index=False)

def generate_table_3_2(df):
    logging.info("Generating Table 3-2: Pathology Distribution")
    
    rows = []
    
    # Steatosis
    s_counts = df['Steatosis_Grade'].value_counts().sort_index()
    total = len(df)
    for grade, count in s_counts.items():
        rows.append({
            'Pathology': 'Steatosis',
            'Grade/Stage': f"S{int(grade)}",
            'Count': count,
            'Percentage': f"{(count/total)*100:.1f}%"
        })
        
    # Fibrosis
    f_counts = df['Fibrosis_Stage'].value_counts().sort_index()
    for stage, count in f_counts.items():
        rows.append({
            'Pathology': 'Fibrosis',
            'Grade/Stage': f"F{int(stage)}",
            'Count': count,
            'Percentage': f"{(count/total)*100:.1f}%"
        })
        
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "task_17.1_table_3_2_pathology.csv"), index=False)

def generate_table_4_1(df):
    logging.info("Generating Table 4-1: Radiomics Features List")
    fusion_df = get_fusion_cohort(df)
    
    # Select radiomics features
    rad_cols = [c for c in fusion_df.columns if c.startswith('original_')]
    
    # Calculate Variance
    variances = fusion_df[rad_cols].var().sort_values(ascending=False)
    top_20 = variances.head(20)
    
    rows = []
    for feat, var in top_20.items():
        # Parse class
        parts = feat.split('_')
        # e.g. original_glcm_Contrast -> Class=glcm, Name=Contrast
        if len(parts) >= 3:
            f_class = parts[1]
            f_name = "_".join(parts[2:])
        else:
            f_class = "FirstOrder"
            f_name = feat
            
        rows.append({
            'Rank': len(rows) + 1,
            'Feature Name': f_name,
            'Feature Class': f_class.upper(),
            'Variance': f"{var:.2f}"
        })
        
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "task_17.1_table_4_1_radiomics.csv"), index=False)

def generate_table_4_2():
    logging.info("Generating Table 4-2: Multi-Target Performance")
    if not os.path.exists(MODEL_LEADERBOARD):
        logging.warning("Model Leaderboard not found!")
        return
        
    df = pd.read_csv(MODEL_LEADERBOARD)
    # Pivot: Index=Target, Columns=Model, Values=AUC_Mean
    pivot = df.pivot(index='Target', columns='Model', values='AUC_Mean')
    
    # Add Std if possible, or just format
    # For a clean table, let's just create a formatted string "Mean ± Std"
    
    formatted_rows = []
    targets = df['Target'].unique()
    models = df['Model'].unique()
    
    for t in targets:
        row = {'Target': t}
        for m in models:
            subset = df[(df['Target'] == t) & (df['Model'] == m)]
            if not subset.empty:
                mean = subset['AUC_Mean'].values[0]
                std = subset['AUC_Std'].values[0]
                row[m] = f"{mean:.3f} ± {std:.3f}"
            else:
                row[m] = "-"
        formatted_rows.append(row)
        
    pd.DataFrame(formatted_rows).to_csv(os.path.join(OUTPUT_DIR, "task_17.1_table_4_2_multi_target.csv"), index=False)

def generate_table_4_3(df):
    logging.info("Generating Table 4-3: Champion Model Performance (Thresholds)")
    fusion_df = get_fusion_cohort(df)
    
    # Targets and Models (Champions)
    # S_Sev -> RF
    # F_Cirr -> RF
    
    tasks = [
        {'name': 'S_Sev', 'col': 'Steatosis_Grade', 'val': 3},
        {'name': 'F_Cirr', 'col': 'Fibrosis_Stage', 'val': 4}
    ]
    
    # Features
    feats = [c for c in fusion_df.columns if c.startswith('original_') or c.endswith('_R') or c.startswith('Index_')]
    feats += ['age', 'BMI', 'T2DM', 'L_S_Ratio', 'Muscle_Mean_HU', 'VAT_SAT_Ratio']
    feats = [c for c in feats if c in fusion_df.columns]
    
    X = fusion_df[feats].values
    
    rows = []
    
    for task in tasks:
        y = (fusion_df[task['col']] == task['val']).astype(int).values
        
        # 5-Fold CV Predictions
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_prob_all = np.zeros(len(y))
        y_true_all = np.zeros(len(y)) # To align order if needed, but array indexing works
        
        # We need to collect all test fold predictions
        # Since we use stratified k-fold, every sample appears exactly once in test set.
        # We can fill an array aligned with original X
        
        # Pipeline
        pipe = ImbPipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42, k_neighbors=min(np.sum(y)-1, 5))),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            pipe.fit(X_train, y_train)
            probs = pipe.predict_proba(X_test)[:, 1]
            y_prob_all[test_idx] = probs
            
        # Metrics at thresholds
        thresholds = [0.3, 0.5, 0.7]
        for th in thresholds:
            y_pred = (y_prob_all >= th).astype(int)
            
            sens = recall_score(y, y_pred)
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = precision_score(y, y_pred, zero_division=0)
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            f1 = f1_score(y, y_pred)
            
            rows.append({
                'Target': task['name'],
                'Model': 'RandomForest',
                'Threshold': th,
                'Sensitivity': f"{sens:.3f}",
                'Specificity': f"{spec:.3f}",
                'PPV': f"{ppv:.3f}",
                'NPV': f"{npv:.3f}",
                'F1-Score': f"{f1:.3f}"
            })
            
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "task_17.1_table_4_3_champion_thresholds.csv"), index=False)

def generate_table_5_1(df):
    logging.info("Generating Table 5-1: SHAP Importance")
    fusion_df = get_fusion_cohort(df)
    
    tasks = [
        {'name': 'S_Sev', 'col': 'Steatosis_Grade', 'val': 3},
        {'name': 'F_Cirr', 'col': 'Fibrosis_Stage', 'val': 4}
    ]
    
    feats = [c for c in fusion_df.columns if c.startswith('original_') or c.endswith('_R') or c.startswith('Index_')]
    feats += ['age', 'BMI', 'T2DM', 'L_S_Ratio', 'Muscle_Mean_HU', 'VAT_SAT_Ratio']
    feats = [c for c in feats if c in fusion_df.columns]
    
    X = fusion_df[feats].values
    
    rows = []
    
    for task in tasks:
        y = (fusion_df[task['col']] == task['val']).astype(int).values
        
        # Train one model on full dataset for interpretation
        imputer = KNNImputer(n_neighbors=5)
        X_imp = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_imp)
        
        # SMOTE
        smote = SMOTE(random_state=42, k_neighbors=min(np.sum(y)-1, 5))
        X_res, y_res = smote.fit_resample(X_s, y)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_res, y_res)
        
        explainer = shap.TreeExplainer(clf)
        # Using a subset if N is huge, but here N~300 is fine
        shap_vals = explainer.shap_values(X_res, check_additivity=False)
        
        # Handle SHAP output shape
        if isinstance(shap_vals, list):
            shap_vals_pos = shap_vals[1]
        elif len(shap_vals.shape) == 3:
            shap_vals_pos = shap_vals[:, :, 1]
        else:
            shap_vals_pos = shap_vals
            
        mean_abs_shap = np.mean(np.abs(shap_vals_pos), axis=0)
        
        # Top 20
        indices = np.argsort(mean_abs_shap)[::-1][:20]
        
        for idx in indices:
            rows.append({
                'Target': task['name'],
                'Feature': feats[idx],
                'Mean |SHAP|': f"{mean_abs_shap[idx]:.4f}"
            })
            
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "task_17.1_table_5_1_shap.csv"), index=False)

def generate_table_5_2(df):
    logging.info("Generating Table 5-2: Human vs AI")
    # This requires processing the JSON text again
    
    # 1. Load Human Labels
    with open(ORIGINAL_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    human_labels = {}
    for entry in data:
        pid = str(entry.get('patient_id', '')).strip()
        if not pid: continue
        ct_text = str(entry.get('CT', '')).lower()
        us_text = str(entry.get('Bchao', '')).lower()
        
        is_sev_fat = lambda t: ('重度' in t or '中重度' in t) and ('脂肪' in t)
        is_cirr = lambda t: '肝硬化' in t
        
        human_labels[pid] = {
            'Human_CT_S_Sev': int(is_sev_fat(ct_text)),
            'Human_CT_Cirr': int(is_cirr(ct_text)),
            'Human_US_S_Sev': int(is_sev_fat(us_text)),
            'Human_US_Cirr': int(is_cirr(us_text))
        }
        
    # 2. Merge with Fusion DF
    fusion_df = get_fusion_cohort(df)
    fusion_df['patient_id'] = fusion_df['patient_id'].astype(str).str.strip()
    
    h_df = pd.DataFrame.from_dict(human_labels, orient='index').reset_index().rename(columns={'index': 'patient_id'})
    merged_df = pd.merge(fusion_df, h_df, on='patient_id', how='left')
    
    # 3. Get AI Predictions (CV)
    tasks = [
        {'name': 'S_Sev', 'col': 'Steatosis_Grade', 'val': 3, 
         'h_ct': 'Human_CT_S_Sev', 'h_us': 'Human_US_S_Sev'},
        {'name': 'F_Cirr', 'col': 'Fibrosis_Stage', 'val': 4,
         'h_ct': 'Human_CT_Cirr', 'h_us': 'Human_US_Cirr'}
    ]
    
    feats = [c for c in fusion_df.columns if c.startswith('original_') or c.endswith('_R') or c.startswith('Index_')]
    feats += ['age', 'BMI', 'T2DM', 'L_S_Ratio', 'Muscle_Mean_HU', 'VAT_SAT_Ratio']
    feats = [c for c in feats if c in fusion_df.columns]
    
    # Ensure X is aligned with merged_df
    X = merged_df[feats].values
    
    rows = []
    
    for task in tasks:
        y = (merged_df[task['col']] == task['val']).astype(int).values
        
        # AI CV - Manual Loop
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_ai = np.full(len(y), -1) # Initialize with -1 to catch unfilled indices
        
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
            y_pred_ai[test_idx] = preds
            
        # Calculate Metrics
        # AI
        if np.any(y_pred_ai == -1):
            logging.error(f"Error: Some samples were not predicted for {task['name']}")
        
        sens_ai = recall_score(y, y_pred_ai)
        tn, fp, fn, tp = confusion_matrix(y, y_pred_ai).ravel()
        spec_ai = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc_ai = accuracy_score(y, y_pred_ai)
        
        rows.append({
            'Target': task['name'],
            'Method': 'AI Model',
            'Sensitivity': f"{sens_ai:.3f}",
            'Specificity': f"{spec_ai:.3f}",
            'Accuracy': f"{acc_ai:.3f}"
        })
        
        # Human CT
        valid_ct = merged_df[merged_df[task['h_ct']].notna()]
        if not valid_ct.empty:
            y_v = (valid_ct[task['col']] == task['val']).astype(int)
            y_h = valid_ct[task['h_ct']]
            sens_h = recall_score(y_v, y_h)
            tn, fp, fn, tp = confusion_matrix(y_v, y_h).ravel()
            spec_h = tn / (tn + fp) if (tn + fp) > 0 else 0
            acc_h = accuracy_score(y_v, y_h)
            
            rows.append({
                'Target': task['name'],
                'Method': 'Human CT Report',
                'Sensitivity': f"{sens_h:.3f}",
                'Specificity': f"{spec_h:.3f}",
                'Accuracy': f"{acc_h:.3f}"
            })
            
        # Human US
        valid_us = merged_df[merged_df[task['h_us']].notna()]
        if not valid_us.empty:
            y_v = (valid_us[task['col']] == task['val']).astype(int)
            y_h = valid_us[task['h_us']]
            sens_h = recall_score(y_v, y_h)
            tn, fp, fn, tp = confusion_matrix(y_v, y_h).ravel()
            spec_h = tn / (tn + fp) if (tn + fp) > 0 else 0
            acc_h = accuracy_score(y_v, y_h)
            
            rows.append({
                'Target': task['name'],
                'Method': 'Human US Report',
                'Sensitivity': f"{sens_h:.3f}",
                'Specificity': f"{spec_h:.3f}",
                'Accuracy': f"{acc_h:.3f}"
            })
            
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "task_17.1_table_5_2_human_vs_ai.csv"), index=False)

def generate_table_6_1(df):
    logging.info("Generating Table 6-1: Pathology Quantification")
    if not os.path.exists(QUANT_RESULTS):
        logging.warning("Quantification results not found!")
        return
        
    quant_df = pd.read_csv(QUANT_RESULTS)
    
    # Fix column case mismatch if present
    if 'Patient_ID' in quant_df.columns:
        quant_df = quant_df.rename(columns={'Patient_ID': 'patient_id'})

    # Ensure ID string matching
    quant_df['patient_id'] = quant_df['patient_id'].astype(str).str.strip()
    df['patient_id'] = df['patient_id'].astype(str).str.strip()
    
    # Merge to get CT_Liver_HU (Liver_Mean_HU)
    merged = pd.merge(quant_df, df[['patient_id', 'Liver_Mean_HU', 'Steatosis_Grade']], on='patient_id', how='left')
    
    # Sort by Pathologist Grade (which is typically the Steatosis_Grade in Master, or 'Pathologist_Grade' in quant csv if it exists?)
    # In quant csv, it usually has the grade if extracted.
    # Let's use 'Steatosis_Grade' from Master as the "Pathologist Grade".
    
    # Select columns
    # Patient_ID, Pathologist_Grade, AI_Fat_Ratio, CT_Liver_HU
    out_df = merged[['patient_id', 'Steatosis_Grade', 'Fat_Ratio', 'Liver_Mean_HU']].copy()
    out_df.columns = ['Patient_ID', 'Pathologist_Grade', 'AI_Fat_Ratio', 'CT_Liver_HU']
    
    # Sort
    out_df = out_df.sort_values(by=['Pathologist_Grade', 'AI_Fat_Ratio'])
    
    out_df.to_csv(os.path.join(OUTPUT_DIR, "task_17.1_table_6_1_quantification.csv"), index=False)

def main():
    ensure_dir(OUTPUT_DIR)
    
    df = load_data()
    
    generate_table_3_1(df)
    generate_table_3_2(df)
    generate_table_4_1(df)
    generate_table_4_2()
    generate_table_4_3(df)
    generate_table_5_1(df)
    generate_table_5_2(df)
    generate_table_6_1(df)
    
    logging.info("All tables generated successfully in " + OUTPUT_DIR)

if __name__ == "__main__":
    main()
