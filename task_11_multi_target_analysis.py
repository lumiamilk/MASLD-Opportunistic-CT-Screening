import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
INPUT_CSV = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
OUTPUT_HEATMAP = os.path.join(BASE_DIR, "output", "multi_target_heatmap.png")
REPORT_MD = os.path.join(BASE_DIR, "output", "task_11_run_report.md")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path):
    try:
        return pd.read_csv(path, encoding='utf-8-sig')
    except:
        return pd.read_csv(path, encoding='gb18030')

def run_experiment(X, y, scale_pos_weight):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Impute & Scale
        imputer = KNNImputer(n_neighbors=5)
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Select (if high dim)
        if X_train.shape[1] > 20:
            selector = SelectKBest(f_classif, k=20)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
            
        # Train XGB
        clf = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=3, 
            learning_rate=0.1, 
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss', 
            random_state=42,
            n_jobs=1
        )
        clf.fit(X_train, y_train)
        
        # Eval
        try:
            preds = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, preds)
            aucs.append(auc)
        except:
            pass # Handle edge cases
            
    return np.mean(aucs) if aucs else 0.5

def main():
    if not os.path.exists(INPUT_CSV):
        logging.error("Input not found.")
        return

    df = load_data(INPUT_CSV)
    
    # Define Targets
    targets = {
        'F (Steatosis >=2)': (df['Steatosis_Grade'] >= 2).astype(int),
        'I (Inflammation >=2)': (df['Inflammation_Grade'] >= 2).astype(int),
        'S2 (Fibrosis >=2)': (df['Fibrosis_Stage'] >= 2).astype(int),
        'S3 (Fibrosis >=3)': (df['Fibrosis_Stage'] >= 3).astype(int)
    }
    
    # Subset: Must have Radiomics OR BodyComp
    # Using L_S_Ratio as proxy for BodyComp, original_firstorder_Energy for Radiomics
    has_valid_ct = df['L_S_Ratio'].notna() | df['original_firstorder_Energy'].notna()
    df = df[has_valid_ct].copy()
    logging.info(f"Analysis Subset Size: {len(df)}")
    
    # Define Feature Sets
    feat_clinical = [c for c in [
        'age', 'BMI', 'T2DM', 'Index_FIB4', 'Index_APRI', 'Index_TyG', 
        'Index_NLR', 'PLT_R', 'AST_R', 'ALT_R', 'GGT_R'
    ] if c in df.columns]
    
    feat_rad = [c for c in df.columns if c.startswith('original_')]
    
    feat_body = [c for c in [
        'L_S_Ratio', 'VAT_SAT_Ratio', 'Muscle_Mean_HU', 'Liver_Mean_HU', 
        'Spleen_Volume', 'Visceral_Fat_Volume', 'Fat_Volume'
    ] if c in df.columns]
    
    feature_sets = {
        'Clinical': feat_clinical,
        'Radiomics': feat_rad,
        'BodyComp': feat_body,
        'Fusion': feat_clinical + feat_rad + feat_body
    }
    
    # Result Matrix: Rows=Targets, Cols=FeatureSets
    results = pd.DataFrame(index=targets.keys(), columns=feature_sets.keys(), dtype=float)
    
    logging.info("Starting Multi-Target Analysis...")
    
    lines = []
    lines.append("# Task 11: Multi-Target Analysis Report")
    lines.append(f"Generated on: {pd.Timestamp.now()}\n")
    lines.append("## 1. Class Balance (Positive Samples)")
    
    for t_name, y_series in targets.items():
        # Align y with df subset
        y_subset = y_series.loc[df.index]
        pos_count = y_subset.sum()
        neg_count = len(y_subset) - pos_count
        ratio = neg_count / pos_count if pos_count > 0 else 1.0
        
        logging.info(f"Target {t_name}: Pos={pos_count}, Neg={neg_count}, Ratio={ratio:.2f}")
        lines.append(f"*   **{t_name}**: {pos_count}/{len(df)} ({pos_count/len(df)*100:.1f}%)")
        
        for f_name, cols in feature_sets.items():
            if not cols: 
                results.loc[t_name, f_name] = 0.5
                continue
                
            X = df[cols].values
            y = y_subset.values
            
            mean_auc = run_experiment(X, y, ratio)
            results.loc[t_name, f_name] = mean_auc
            logging.info(f"  -> {f_name}: AUC={mean_auc:.3f}")

    # Plot Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(results, annot=True, cmap="RdYlGn", fmt=".3f", vmin=0.5, vmax=0.9)
    plt.title("AUC Performance Heatmap: Target vs Modality")
    plt.savefig(OUTPUT_HEATMAP)
    logging.info(f"Heatmap saved to {OUTPUT_HEATMAP}")
    
    # Report Table
    lines.append("\n## 2. AUC Matrix")
    lines.append(results.to_markdown())
    
    # Delta Analysis
    lines.append("\n## 3. Incremental Value Analysis (Fusion - Clinical)")
    lines.append("| Target | Fusion AUC | Clinical AUC | Delta | Verdict |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    best_target = ""
    max_delta = -1.0
    
    for t_name in targets.keys():
        fusion = results.loc[t_name, 'Fusion']
        clinical = results.loc[t_name, 'Clinical']
        delta = fusion - clinical
        verdict = "👍 Improvement" if delta > 0.02 else ("🔻 Noise" if delta < -0.02 else "Permille")
        lines.append(f"| {t_name} | {fusion:.3f} | {clinical:.3f} | **{delta:+.3f}** | {verdict} |")
        
        if delta > max_delta:
            max_delta = delta
            best_target = t_name
            
    lines.append(f"\n## 4. Conclusion")
    lines.append(f"*   **Best Enhancement**: The Fusion model showed the biggest improvement over Clinical baselines in **{best_target}** (Delta: {max_delta:+.3f}).")
    lines.append("*   This suggests CT features contribute most unique information for this specific phenotype.")
    
    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
