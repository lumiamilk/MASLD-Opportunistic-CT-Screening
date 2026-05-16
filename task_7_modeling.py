import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, confusion_matrix
import xgboost as xgb

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
INPUT_CSV = os.path.join(BASE_DIR, "output", "master_dataset_with_indices.csv")
OUTPUT_ROC = os.path.join(BASE_DIR, "output", "roc_comparison.png")
OUTPUT_FI = os.path.join(BASE_DIR, "output", "feature_importance.csv")
REPORT_MD = os.path.join(BASE_DIR, "output", "task_7_run_report.md")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_report(results, n_samples, n_features, fi_df):
    """
    使用最稳妥的列表追加方式生成报告，避免任何引号嵌套导致的语法错误。
    您可以在此处手动修改报告的文字内容。
    """
    # 获取 Combined 实验的前 10 个重要特征
    top10 = fi_df[fi_df['Experiment'] == 'Combined'].sort_values('Importance', ascending=False).head(10)
    
    lines = []
    lines.append("# Task 7 Modeling Report: Fibrosis Prediction")
    lines.append(f"Generated on: {pd.Timestamp.now()}")
    lines.append("")
    lines.append("## 1. Experiment Setup")
    lines.append(f"*   **Target**: Significant Fibrosis (Stage >= 2)")
    lines.append(f"*   **Sample Size**: {n_samples} (Multimodal Subset)")
    lines.append(f"*   **Total Features**: {n_features}")
    lines.append("*   **Method**: XGBoost Classifier (5-Fold Stratified CV) + Lasso Selection + Median Imputation")
    lines.append("")
    lines.append("## 2. Model Performance (Mean metrics over 5 folds)")
    lines.append("| Experiment | AUC (Mean ± Std) | Accuracy | Sensitivity | Specificity |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    for name, res in results.items():
        line = f"| **{name}** | {res['AUC']:.3f} ± {res['AUC_std']:.3f} | {res['Acc']:.3f} | {res['Sens']:.3f} | {res['Spec']:.3f} |"
        lines.append(line)
        
    lines.append("")
    lines.append("## 3. Top 10 Important Features (Combined Model)")
    lines.append("| Rank | Feature | Importance Score |")
    lines.append("| :--- | :--- | :--- |")

    for i, (idx, row) in enumerate(top10.iterrows(), 1):
        line = f"| {i} | `{row['Feature']}` | {row['Importance']:.4f} |"
        lines.append(line)

    lines.append("")
    lines.append("## 4. Key Insights")
    lines.append("*   **Best Model**: The experiment with the highest AUC indicates the most effective feature set.")
    lines.append("*   **Incremental Value**: Compare 'Combined' vs 'Clinical Only' to see if CT adds value.")
    lines.append("*   **Feature Importance**: Top features reveal the biological or radiological drivers of fibrosis.")
    
    report_content = "\n".join(lines)
    
    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write(report_content)
    logging.info(f"Report saved to {REPORT_MD}")

def main():
    if not os.path.exists(INPUT_CSV):
        logging.error(f"Input file not found: {INPUT_CSV}")
        return

    # 1. Data Preparation
    try:
        df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
    except:
        df = pd.read_csv(INPUT_CSV, encoding='gb18030')
        
    logging.info(f"Loaded {len(df)} records.")
    
    # Define Target: Significant Fibrosis (S >= 2)
    df = df[df['Fibrosis_Stage'].notna()].copy()
    df['Label_Fibrosis_Sig'] = (df['Fibrosis_Stage'] >= 2).astype(int)
    
    # Check for radiomics column
    rad_col_check = 'original_firstorder_Energy'
    if rad_col_check not in df.columns:
        logging.error("Radiomics features not found!")
        return

    df_fusion = df[df[rad_col_check].notna()].copy()
    logging.info(f"Fusion Subset Size (N): {len(df_fusion)}")

    # 2. Feature Definition
    feat_clinical = [
        'age', 'sex', 'BMI', 'T2DM', 'High_Blood_pressure',
        'Index_FIB4', 'Index_APRI', 'Index_TyG', 'Index_TyG_BMI', 
        'Index_NLR', 'Index_PLR', 'Index_SII', 'Index_AST_ALT',
        'ALT_R', 'AST_R', 'PLT_R', 'GGT_R'
    ]
    feat_clinical = [c for c in feat_clinical if c in df_fusion.columns]
    feat_radiomics = [c for c in df_fusion.columns if c.startswith('original_')]
    feat_combined = feat_clinical + feat_radiomics
    
    # 3. Experiment Loop
    experiments = {
        'Clinical Only': feat_clinical,
        'Radiomics Only': feat_radiomics,
        'Combined': feat_combined
    }
    
    results = {}
    plt.figure(figsize=(10, 8))
    global_fi = []

    for name, features in experiments.items():
        logging.info(f"Running Experiment: {name}")
        X = df_fusion[features].values
        y = df_fusion['Label_Fibrosis_Sig'].values
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        tprs, aucs = [], []
        mean_fpr = np.linspace(0, 1, 100)
        metrics = {'acc': [], 'sens': [], 'spec': []}
        fold_fi = np.zeros(len(features))
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Preprocessing
            imputer = SimpleImputer(strategy='median')
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Selection for high-dim
            if X_train.shape[1] > 20:
                # Use SelectKBest (ANOVA F-value) for stability instead of Lasso
                # This guarantees we always select k features and avoids convergence issues
                selector = SelectKBest(f_classif, k=20)
                X_train_sel = selector.fit_transform(X_train, y_train)
                X_test_sel = selector.transform(X_test)
                support = selector.get_support()
            else:
                X_train_sel, X_test_sel = X_train, X_test
                support = np.ones(X_train.shape[1], dtype=bool)

            clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric='logloss', random_state=42)
            clf.fit(X_train_sel, y_train)
            
            y_prob = clf.predict_proba(X_test_sel)[:, 1]
            y_pred = clf.predict(X_test_sel)
            
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(auc(fpr, tpr))
            metrics['acc'].append(accuracy_score(y_test, y_pred))
            metrics['sens'].append(recall_score(y_test, y_pred))
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            metrics['spec'].append(tn / (tn + fp))
            
            if hasattr(clf, 'feature_importances_'):
                importance_map = np.zeros(len(features))
                importance_map[support] += clf.feature_importances_
                fold_fi += importance_map

        results[name] = {'AUC': np.mean(aucs), 'AUC_std': np.std(aucs), 'Acc': np.mean(metrics['acc']), 'Sens': np.mean(metrics['sens']), 'Spec': np.mean(metrics['spec'])}
        plt.plot(mean_fpr, np.mean(tprs, axis=0), label=f"{name} (AUC = {np.mean(aucs):.2f})")
        global_fi.append(pd.DataFrame({'Feature': features, 'Importance': fold_fi/5.0, 'Experiment': name}))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.savefig(OUTPUT_ROC)
    pd.concat(global_fi).to_csv(OUTPUT_FI, index=False)
    generate_report(results, len(df_fusion), len(feat_combined), pd.concat(global_fi))

if __name__ == "__main__":
    main()
