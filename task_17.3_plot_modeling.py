import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibrationDisplay
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import logging

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
MASTER_CSV = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
TABLE_4_2_CSV = os.path.join(BASE_DIR, "output", "thesis_assets", "tables", "task_17.1_table_4_2_multi_target.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "thesis_assets", "figures")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300

def load_data():
    try: df = pd.read_csv(MASTER_CSV, encoding='utf-8-sig')
    except: df = pd.read_csv(MASTER_CSV, encoding='gb18030')
    
    # Fusion Cohort only
    mask = df['L_S_Ratio'].notna() | df['original_firstorder_Energy'].notna()
    return df[mask].copy()

def save_fig(name):
    path_png = os.path.join(OUTPUT_DIR, f"{name}.png")
    path_tif = os.path.join(OUTPUT_DIR, f"{name}.tif")
    plt.savefig(path_png, bbox_inches='tight', dpi=300)
    plt.savefig(path_tif, bbox_inches='tight', dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    logging.info(f"Saved {name}")
    plt.close()

# --- Task 1: Multi-Target AUC Heatmap ---
def fig_4_1_heatmap():
    if not os.path.exists(TABLE_4_2_CSV):
        logging.warning("Table 4-2 not found, skipping Fig 4-1")
        return

    df = pd.read_csv(TABLE_4_2_CSV)
    
    # Data is like: Target | LR | SVM ... |
    # Values are "0.850 ± 0.020"
    # We need to extract the Mean.
    
    df_plot = df.set_index('Target')
    
    def extract_mean(val):
        if isinstance(val, str) and '±' in val:
            return float(val.split('±')[0].strip())
        try:
            return float(val)
        except:
            return np.nan
            
    df_numeric = df_plot.applymap(extract_mean)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_numeric, annot=True, cmap="RdYlGn", fmt=".3f", 
                vmin=0.5, vmax=0.95, cbar_kws={'label': 'AUC (Mean)'})
    plt.title("Fig 4-1: Multi-Target Model Performance (AUC)")
    plt.xlabel("Machine Learning Model")
    plt.ylabel("Clinical Target")
    save_fig("Fig_4_1_Multi_Target_Heatmap")

# --- Helper: Model Training for Plots ---
def get_feature_sets(df):
    rad_cols = [c for c in df.columns if c.startswith('original_')]
    
    # Check exact column names from master csv head
    clinical_cols = ['age', 'sex', 'BMI', 'T2DM', 'High_Blood_pressure', 'ALT_Val', 'AST_Val', 'PLT_Val', 'TG_Val']
    body_cols = ['Liver_Mean_HU', 'L_S_Ratio', 'Muscle_Mean_HU', 'Fat_Volume', 'Visceral_Fat_Volume', 'Spleen_Volume', 'VAT_SAT_Ratio']
    
    # Filter existing
    clinical_cols = [c for c in clinical_cols if c in df.columns]
    body_cols = [c for c in body_cols if c in df.columns]
    
    sets = {
        'Clinical': clinical_cols,
        'Radiomics': rad_cols,
        'BodyComp': body_cols,
        'Fusion': clinical_cols + rad_cols + body_cols
    }
    return sets

def train_and_get_probs(df, feature_cols, target_col, target_val=1):
    X = df[feature_cols].values
    y = (df[target_col] == target_val).astype(int).values
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all = []
    y_prob_all = []
    
    # For mean ROC
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Pipeline
        # SMOTE k_neighbors check
        k = min(np.sum(y_train)-1, 5)
        if k < 1: k = 1
        
        pipe = ImbPipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42, k_neighbors=k)),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        
        y_true_all.extend(y_test)
        y_prob_all.extend(probs)
        
        fpr, tpr, _ = roc_curve(y_test, probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
        
    return np.array(y_true_all), np.array(y_prob_all), tprs, aucs, mean_fpr

# --- Task 2 & 3: ROC Curves ---
def fig_roc_comparison(df, target_col, target_val, title_suffix, fig_name):
    feature_sets = get_feature_sets(df)
    
    plt.figure(figsize=(8, 8))
    colors = {'Clinical': 'blue', 'Radiomics': 'green', 'BodyComp': 'orange', 'Fusion': 'red'}
    
    for name, cols in feature_sets.items():
        if not cols: continue
        
        _, _, tprs, aucs, mean_fpr = train_and_get_probs(df, cols, target_col, target_val)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        plt.plot(mean_fpr, mean_tpr, color=colors.get(name, 'black'),
                 label=f'{name} (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})', lw=2, alpha=0.8)
                 
        # Std Deviation Area (Optional, might be too messy with 4 curves)
        # std_tpr = np.std(tprs, axis=0)
        # tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        # plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=colors.get(name, 'black'), alpha=0.1)
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {title_suffix}')
    plt.legend(loc="lower right")
    save_fig(fig_name)

# --- Task 4: Calibration Curves ---
def fig_4_4_calibration(df):
    feature_sets = get_feature_sets(df)
    cols = feature_sets['Fusion']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: S_Sev
    y_true, y_prob, _, _, _ = train_and_get_probs(df, cols, 'Steatosis_Grade', 3)
    display = CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=5, ax=axes[0], name='Fusion Model')
    axes[0].set_title("Calibration: Severe Steatosis (S=3)")
    
    # Subplot 2: F_Cirr
    y_true, y_prob, _, _, _ = train_and_get_probs(df, cols, 'Fibrosis_Stage', 4)
    display = CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=5, ax=axes[1], name='Fusion Model')
    axes[1].set_title("Calibration: Cirrhosis (F=4)")
    
    save_fig("Fig_4_4_Calibration_Curves")

# --- Task 5: Decision Curve Analysis ---
def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefits = []
    n = len(y_true)
    
    for pt in thresholds:
        # Treat if prob >= pt
        y_pred = (y_prob >= pt).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        if pt == 1.0:
            nb = 0
        else:
            weight = pt / (1 - pt)
            nb = (tp / n) - (fp / n) * weight
            
        net_benefits.append(nb)
        
    return np.array(net_benefits)

def fig_4_5_dca(df):
    # Focus on S_Sev
    target_col = 'Steatosis_Grade'
    target_val = 3
    
    feature_sets = get_feature_sets(df)
    
    # Get probs for Fusion and Clinical
    y_true, prob_fusion, _, _, _ = train_and_get_probs(df, feature_sets['Fusion'], target_col, target_val)
    _, prob_clinical, _, _, _ = train_and_get_probs(df, feature_sets['Clinical'], target_col, target_val)
    
    thresholds = np.linspace(0.01, 0.99, 99)
    
    nb_fusion = calculate_net_benefit(y_true, prob_fusion, thresholds)
    nb_clinical = calculate_net_benefit(y_true, prob_clinical, thresholds)
    
    # Treat All: prob is always 1 (>= threshold) for low thresholds? 
    # Actually Treat All means predicting 1 for everyone.
    # NB_all = (Prevalence) - (1-Prevalence) * (pt / 1-pt)
    prevalence = np.sum(y_true) / len(y_true)
    nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    
    # Treat None: NB = 0
    nb_none = np.zeros_like(thresholds)
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, nb_fusion, label='Fusion Model', color='red', lw=2)
    plt.plot(thresholds, nb_clinical, label='Clinical Model', color='blue', lw=2, linestyle='--')
    plt.plot(thresholds, nb_all, label='Treat All', color='gray', linestyle=':')
    plt.plot(thresholds, nb_none, label='Treat None', color='black', lw=1)
    
    # Limit y-axis to reasonable range (often -0.05 to max benefit)
    y_max = max(np.max(nb_fusion), np.max(nb_clinical), prevalence) + 0.05
    y_min = -0.05 # Standard in DCA plots
    plt.ylim(y_min, y_max)
    plt.xlim(0, 1.0)
    
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Fig 4-5: Decision Curve Analysis (Severe Steatosis)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    save_fig("Fig_4_5_DCA_S_Sev")

def main():
    ensure_dir(OUTPUT_DIR)
    set_style()
    df = load_data()
    
    fig_4_1_heatmap()
    
    # S_Sev
    fig_roc_comparison(df, 'Steatosis_Grade', 3, 'Severe Steatosis (S=3)', 'Fig_4_2_ROC_S_Sev')
    
    # F_Cirr
    fig_roc_comparison(df, 'Fibrosis_Stage', 4, 'Cirrhosis (F=4)', 'Fig_4_3_ROC_F_Cirr')
    
    fig_4_4_calibration(df)
    fig_4_5_dca(df)
    
    logging.info("All Modeling Figures Generated.")

if __name__ == "__main__":
    main()
