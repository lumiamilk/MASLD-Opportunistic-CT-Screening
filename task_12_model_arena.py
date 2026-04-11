import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Pipeline & Eval
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
INPUT_CSV = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
OUTPUT_HEATMAP = os.path.join(BASE_DIR, "output", "model_arena_heatmap.png")
OUTPUT_LEADERBOARD = os.path.join(BASE_DIR, "output", "model_arena_leaderboard.csv")
REPORT_MD = os.path.join(BASE_DIR, "output", "task_12_run_report.md")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path):
    try: return pd.read_csv(path, encoding='utf-8-sig')
    except: return pd.read_csv(path, encoding='gb18030')

def get_model_zoo():
    return {
        'LR': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGB': xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric='logloss', random_state=42, n_jobs=1),
        'LGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=-1, random_state=42, n_jobs=1),
        'CAT': CatBoostClassifier(iterations=100, depth=3, learning_rate=0.1, verbose=0, random_state=42, allow_writing_files=False),
        'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    }

def main():
    if not os.path.exists(INPUT_CSV): return

    df = load_data(INPUT_CSV)
    
    # Subset: Fusion ready
    has_valid_ct = df['L_S_Ratio'].notna() | df['original_firstorder_Energy'].notna()
    df = df[has_valid_ct].copy()
    logging.info(f"Arena Subset: {len(df)}")
    
    # Targets
    targets = {
        'S_Mild (S>=1)': (df['Steatosis_Grade'] >= 1).astype(int),
        'S_Mod (S>=2)': (df['Steatosis_Grade'] >= 2).astype(int),
        'S_Sev (S=3)': (df['Steatosis_Grade'] == 3).astype(int),
        'F_Adv (F>=3)': (df['Fibrosis_Stage'] >= 3).astype(int),
        'F_Cirr (F=4)': (df['Fibrosis_Stage'] == 4).astype(int)
    }
    
    # Features (Fusion Set)
    feat_cols = [c for c in df.columns if c.startswith('original_') or c.endswith('_R') or c.startswith('Index_')]
    feat_cols += ['age', 'BMI', 'T2DM', 'L_S_Ratio', 'Muscle_Mean_HU', 'VAT_SAT_Ratio']
    feat_cols = [c for c in feat_cols if c in df.columns]
    
    # Clean numerical features
    X_raw = df[feat_cols].values
    
    model_zoo = get_model_zoo()
    
    results = [] # List of dicts
    
    logging.info("--- Entering The Arena ---")
    
    for t_name, y_raw in targets.items():
        pos_count = y_raw.sum()
        if pos_count < 10:
            logging.warning(f"Skipping {t_name}: Too few positives ({pos_count})")
            continue
            
        logging.info(f"Target: {t_name} (Pos={pos_count})")
        
        for m_name, model in model_zoo.items():
            
            # Pipeline Construction
            # 1. Impute (KNN)
            # 2. Scale
            # 3. SMOTE (Oversampling) - Crucial for F_Cirr
            # 4. Select (Top 25)
            # 5. Model
            
            # Note: MLP needs dense data, XGB/LGBM handle NaN but Pipeline makes it uniform
            pipe = ImbPipeline([
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=42, k_neighbors=min(pos_count-1, 5))),
                ('select', SelectKBest(f_classif, k=25)),
                ('clf', model)
            ])
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            aucs = []
            
            for train_idx, test_idx in cv.split(X_raw, y_raw):
                X_train, X_test = X_raw[train_idx], X_raw[test_idx]
                y_train, y_test = y_raw.values[train_idx], y_raw.values[test_idx]
                
                try:
                    pipe.fit(X_train, y_train)
                    # Get probabilities
                    if hasattr(pipe['clf'], "predict_proba"):
                        preds = pipe.predict_proba(X_test)[:, 1]
                    else:
                        preds = pipe.decision_function(X_test)
                        
                    auc = roc_auc_score(y_test, preds)
                    aucs.append(auc)
                except Exception as e:
                    # logging.error(f"Error in {t_name}-{m_name}: {e}")
                    pass
            
            mean_auc = np.mean(aucs) if aucs else 0.5
            std_auc = np.std(aucs) if aucs else 0.0
            
            results.append({
                'Target': t_name,
                'Model': m_name,
                'AUC_Mean': mean_auc,
                'AUC_Std': std_auc
            })
            # logging.info(f"  > {m_name}: {mean_auc:.3f}")

    # Output
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_LEADERBOARD, index=False)
    
    # Pivot for Heatmap
    heatmap_data = res_df.pivot(index='Target', columns='Model', values='AUC_Mean')
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="RdYlGn", fmt=".3f", vmin=0.5, vmax=0.95)
    plt.title("Model Arena: AUC Performance (Fusion Features + SMOTE)")
    plt.tight_layout()
    plt.savefig(OUTPUT_HEATMAP)
    logging.info(f"Heatmap saved: {OUTPUT_HEATMAP}")
    
    # Report Generation
    generate_report(res_df)

def generate_report(df):
    # Find Champion for each target
    champions = df.loc[df.groupby('Target')['AUC_Mean'].idxmax()]
    
    lines = []
    lines.append("# Task 12 Report: Model Arena & Multi-Threshold Analysis")
    lines.append(f"Generated on: {pd.Timestamp.now()}")
    lines.append("")
    lines.append("## 1. Champion Models (Best per Target)")
    lines.append("| Target | Champion Model | AUC (Mean ± Std) |")
    lines.append("| :--- | :--- | :--- |")
    
    for _, row in champions.iterrows():
        lines.append(f"| **{row['Target']}** | `{row['Model']}` | **{row['AUC_Mean']:.3f}** ± {row['AUC_Std']:.3f} |")
        
    lines.append("")
    lines.append("## 2. Leaderboard Summary")
    lines.append("Models ranked by Average AUC across all targets:")
    avg_perf = df.groupby('Model')['AUC_Mean'].mean().sort_values(ascending=False)
    
    lines.append("| Rank | Model | Avg AUC |")
    lines.append("| :--- | :--- | :--- |")
    for i, (m, score) in enumerate(avg_perf.items(), 1):
        lines.append(f"| {i} | {m} | {score:.3f} |")
        
    lines.append("")
    lines.append("## 3. Analysis")
    lines.append("*   **SMOTE Effect**: Handling imbalance is crucial for 'Sev' and 'Cirr' targets.")
    lines.append("*   **Fat vs Fibrosis**: Observe the AUC gap between 'S' targets and 'F' targets.")
    lines.append("*   **Algorithm**: Tree-based models (CAT, XGB, LGBM) vs Linear (LR) - who wins?")
    
    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    logging.info(f"Report saved: {REPORT_MD}")

if __name__ == "__main__":
    main()
