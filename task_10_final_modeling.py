import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, confusion_matrix

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
INPUT_CSV = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
OUTPUT_ROC = os.path.join(BASE_DIR, "output", "final_roc_comparison.png")
OUTPUT_SHAP = os.path.join(BASE_DIR, "output", "shap_summary.png")
REPORT_MD = os.path.join(BASE_DIR, "output", "task_10_run_report.md")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path):
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(path, encoding='gb18030')
    return df

def main():
    if not os.path.exists(INPUT_CSV):
        logging.error("Input missing.")
        return

    df = load_data(INPUT_CSV)
    
    # Target
    df = df[df['Fibrosis_Stage'].notna()].copy()
    df['Label_Fibrosis_Sig'] = (df['Fibrosis_Stage'] >= 2).astype(int)
    
    # Subset (Must have CT)
    # Using L_S_Ratio as indicator of successful CT analysis (Task 8)
    # OR original_firstorder_Energy (Task 4)
    # We take the union of validity
    has_rad = df['original_firstorder_Energy'].notna()
    has_body = df['L_S_Ratio'].notna()
    df_fusion = df[has_rad | has_body].copy()
    
    logging.info(f"Total Samples: {len(df_fusion)}")
    
    # Features
    feat_clinical = [
        'age', 'BMI', 'T2DM', 'Index_FIB4', 'Index_APRI', 'Index_TyG', 
        'Index_NLR', 'PLT_R', 'AST_R', 'ALT_R'
    ]
    
    feat_radiomics = [c for c in df_fusion.columns if c.startswith('original_')]
    
    feat_body = [
        'L_S_Ratio', 'VAT_SAT_Ratio', 'Muscle_Mean_HU', 'Liver_Mean_HU', 
        'Spleen_Volume', 'Visceral_Fat_Volume', 'Fat_Volume'
    ]
    
    # Filter available
    feat_clinical = [c for c in feat_clinical if c in df_fusion.columns]
    feat_radiomics = [c for c in feat_radiomics if c in df_fusion.columns]
    feat_body = [c for c in feat_body if c in df_fusion.columns]
    
    # Experiments
    experiments = {
        'Clinical Only': feat_clinical,
        'Radiomics Only': feat_radiomics,
        'BodyComp Only': feat_body,
        'Fusion (All)': feat_clinical + feat_body + feat_radiomics
    }
    
    results = {}
    plt.figure(figsize=(10, 8))
    
    # Prepare X, y
    y = df_fusion['Label_Fibrosis_Sig'].values
    
    for name, feats in experiments.items():
        logging.info(f"Experiment: {name} ({len(feats)} feats)")
        X = df_fusion[feats].values
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs, tprs = [], []
        mean_fpr = np.linspace(0, 1, 100)
        metrics = {'acc': [], 'sens': [], 'spec': []}
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Impute & Scale
            imputer = SimpleImputer(strategy='median')
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Select
            if X_train.shape[1] > 20:
                sel = SelectKBest(f_classif, k=20)
                X_train = sel.fit_transform(X_train, y_train)
                X_test = sel.transform(X_test)
            
            # Train
            clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric='logloss', random_state=42)
            clf.fit(X_train, y_train)
            
            # Eval
            probs = clf.predict_proba(X_test)[:, 1]
            preds = clf.predict(X_test)
            
            fpr, tpr, _ = roc_curve(y_test, probs)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(auc(fpr, tpr))
            metrics['acc'].append(accuracy_score(y_test, preds))
            metrics['sens'].append(recall_score(y_test, preds))
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            metrics['spec'].append(tn / (tn + fp))
            
        results[name] = {
            'AUC': np.mean(aucs), 'Std': np.std(aucs), 
            'Acc': np.mean(metrics['acc']), 
            'Sens': np.mean(metrics['sens']), 
            'Spec': np.mean(metrics['spec'])
        }
        plt.plot(mean_fpr, np.mean(tprs, axis=0), label=f"{name} (AUC={np.mean(aucs):.3f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title('Final ROC Comparison')
    plt.savefig(OUTPUT_ROC)
    logging.info("ROC Saved.")
    
    # SHAP Analysis on Fusion Model (Full Data)
    logging.info("Running SHAP analysis on Fusion model...")
    feats = experiments['Fusion (All)']
    X_full = df_fusion[feats].values
    y_full = df_fusion['Label_Fibrosis_Sig'].values
    
    # Impute first (SHAP needs clean data, though XGB handles NaN, scaler needs it)
    imputer = SimpleImputer(strategy='median')
    X_full = imputer.fit_transform(X_full)
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)
    
    # Train final model
    clf_final = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    clf_final.fit(X_full_scaled, y_full)
    
    # Run SHAP
    explainer = shap.TreeExplainer(clf_final)
    shap_values = explainer.shap_values(X_full_scaled)
    
    plt.figure()
    shap.summary_plot(shap_values, X_full_scaled, feature_names=feats, show=False, max_display=15)
    plt.savefig(OUTPUT_SHAP, bbox_inches='tight')
    logging.info("SHAP Saved.")
    
    # Generate Report
    generate_report(results, feats, clf_final, X_full_scaled)

def generate_report(results, features, clf, X):
    lines = []
    lines.append("# Task 10 Final Report: Multimodal Modeling & Interpretability")
    lines.append(f"Generated on: {pd.Timestamp.now()}")
    lines.append("")
    lines.append("## 1. Model Performance (5-Fold CV)")
    lines.append("| Experiment | AUC | Accuracy | Sensitivity | Specificity |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for k, v in results.items():
        lines.append(f"| **{k}** | {v['AUC']:.3f} ± {v['Std']:.3f} | {v['Acc']:.3f} | {v['Sens']:.3f} | {v['Spec']:.3f} |")
        
    lines.append("")
    lines.append("## 2. SHAP Feature Importance (Fusion Model)")
    # Get top features from xgboost gain just for listing
    imp = clf.feature_importances_
    indices = np.argsort(imp)[::-1][:10]
    
    lines.append("| Rank | Feature | Importance (Gain) |")
    lines.append("| :--- | :--- | :--- |")
    for i, idx in enumerate(indices, 1):
        lines.append(f"| {i} | `{features[idx]}` | {imp[idx]:.4f} |")
        
    lines.append("")
    lines.append("## 3. Conclusion")
    lines.append("*   **BodyComp Value**: Check if 'BodyComp Only' outperforms 'Radiomics Only'.")
    lines.append("*   **Clinical Dominance**: Clinical features usually dominate fibrosis prediction.")
    lines.append("*   **SHAP Analysis**: The beeswarm plot (`shap_summary.png`) visualizes the directionality of impact.")
    
    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
