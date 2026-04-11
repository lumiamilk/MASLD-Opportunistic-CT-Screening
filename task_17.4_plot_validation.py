import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import logging

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
MASTER_CSV = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
QUANT_RESULTS_CSV = os.path.join(BASE_DIR, "output", "task_16_quant_results_v5.3.csv")
TABLE_5_2_CSV = os.path.join(BASE_DIR, "output", "thesis_assets", "tables", "task_17.1_table_5_2_human_vs_ai.csv")
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

def load_master():
    try: df = pd.read_csv(MASTER_CSV, encoding='utf-8-sig')
    except: df = pd.read_csv(MASTER_CSV, encoding='gb18030')
    mask = df['L_S_Ratio'].notna() | df['original_firstorder_Energy'].notna()
    return df[mask].copy()

def save_fig(name):
    path_png = os.path.join(OUTPUT_DIR, f"{name}.png")
    path_tif = os.path.join(OUTPUT_DIR, f"{name}.tif")
    plt.savefig(path_png, bbox_inches='tight', dpi=300)
    plt.savefig(path_tif, bbox_inches='tight', dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    logging.info(f"Saved {name}")
    plt.close()

# --- SHAP Helper ---
def run_shap_analysis(df, target_col, target_val):
    # Features
    feats = [c for c in df.columns if c.startswith('original_') or c.endswith('_R') or c.startswith('Index_')]
    feats += ['age', 'BMI', 'T2DM', 'L_S_Ratio', 'Muscle_Mean_HU', 'VAT_SAT_Ratio', 'Liver_Mean_HU', 'Spleen_Volume', 'Visceral_Fat_Volume']
    feats = [c for c in feats if c in df.columns]
    
    # Clean X
    X = df[feats].values
    y = (df[target_col] == target_val).astype(int).values
    
    # Pipeline steps manually
    imputer = KNNImputer(n_neighbors=5)
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_imp)
    
    # SMOTE
    k = min(np.sum(y)-1, 5)
    if k > 0:
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_s, y)
    else:
        X_res, y_res = X_s, y
        
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_res, y_res)
    
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_res, check_additivity=False)
    
    # Get positive class SHAP
    if isinstance(shap_values, list):
        shap_vals_pos = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_vals_pos = shap_values[:, :, 1]
    else:
        shap_vals_pos = shap_values
        
    return shap_vals_pos, X_res, feats

# --- Task 1: SHAP Beeswarm ---
def fig_5_1_2_shap(df):
    # Fig 5-1: S_Sev
    shap_vals, X_res, feats = run_shap_analysis(df, 'Steatosis_Grade', 3)
    
    plt.figure()
    shap.summary_plot(shap_vals, X_res, feature_names=feats, max_display=15, show=False)
    plt.title("Fig 5-1: SHAP Summary (Severe Steatosis)")
    save_fig("Fig_5_1_SHAP_S_Sev")
    
    # Fig 5-2: F_Cirr
    shap_vals, X_res, feats = run_shap_analysis(df, 'Fibrosis_Stage', 4)
    
    plt.figure()
    shap.summary_plot(shap_vals, X_res, feature_names=feats, max_display=15, show=False)
    plt.title("Fig 5-2: SHAP Summary (Cirrhosis)")
    save_fig("Fig_5_2_SHAP_F_Cirr")
    
    return shap_vals, X_res, feats # Return F_Cirr data for 5-3

# --- Task 2: Dependence Plot ---
def fig_5_3_dependence(shap_vals, X_res, feats):
    # Find Muscle_Mean_HU index
    target_feat = 'Muscle_Mean_HU'
    if target_feat not in feats:
        logging.warning(f"{target_feat} not found in features, skipping Fig 5-3")
        return

    idx = feats.index(target_feat)
    
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(idx, shap_vals, X_res, feature_names=feats, show=False)
    plt.title(f"Fig 5-3: SHAP Dependence ({target_feat} vs Risk)")
    # dependence_plot modifies current axis or figure.
    # It usually creates a scatter plot.
    
    # We need to save the current figure.
    save_fig("Fig_5_3_SHAP_Dependence_Muscle")

# --- Task 3: Human vs AI Bar Chart ---
def fig_5_4_human_vs_ai():
    if not os.path.exists(TABLE_5_2_CSV):
        return
        
    df = pd.read_csv(TABLE_5_2_CSV)
    
    plt.figure(figsize=(10, 6))
    # Bar plot: X=Target, Y=Sensitivity, Hue=Method
    sns.barplot(data=df, x='Target', y='Sensitivity', hue='Method', palette='Set2')
    
    plt.title("Fig 5-4: Sensitivity Comparison (Human vs AI)")
    plt.ylabel("Sensitivity (Recall)")
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    save_fig("Fig_5_4_Human_vs_AI_Sensitivity")

# --- Task 4 & 5: Pathology Correlation ---
def fig_6_2_3_pathology_corr(master_df):
    if not os.path.exists(QUANT_RESULTS_CSV):
        return
        
    quant = pd.read_csv(QUANT_RESULTS_CSV)
    # Fix ID
    if 'Patient_ID' in quant.columns:
        quant = quant.rename(columns={'Patient_ID': 'patient_id'})
    quant['patient_id'] = quant['patient_id'].astype(str).str.strip()
    master_df['patient_id'] = master_df['patient_id'].astype(str).str.strip()
    
    merged = pd.merge(quant, master_df[['patient_id', 'Steatosis_Grade', 'Liver_Mean_HU']], on='patient_id')
    
    if merged.empty:
        logging.warning("Merged pathology data is empty!")
        return
        
    # Fig 6-2: Scatter
    plt.figure(figsize=(8, 6))
    sns.regplot(data=merged, x='Fat_Ratio', y='Liver_Mean_HU', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    
    # Stats
    r, p = stats.pearsonr(merged['Fat_Ratio'], merged['Liver_Mean_HU'])
    plt.text(0.05, 0.95, f"Pearson r = {r:.3f}\np < 0.001", transform=plt.gca().transAxes, 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
             
    plt.xlabel("AI Pathological Fat Ratio")
    plt.ylabel("CT Liver Density (HU)")
    plt.title("Fig 6-2: Correlation (Pathology vs CT)")
    save_fig("Fig_6_2_Pathology_CT_Correlation")
    
    # Fig 6-3: Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=merged, x='Steatosis_Grade', y='Fat_Ratio', palette='viridis', showfliers=False)
    sns.stripplot(data=merged, x='Steatosis_Grade', y='Fat_Ratio', color='black', alpha=0.3)
    
    # Add ANOVA P
    groups = [merged[merged['Steatosis_Grade']==g]['Fat_Ratio'].values for g in sorted(merged['Steatosis_Grade'].unique())]
    if len(groups) > 1:
        _, p = stats.f_oneway(*groups)
        plt.text(0.05, 0.95, f"ANOVA p < 0.001" if p < 0.001 else f"p={p:.3f}", transform=plt.gca().transAxes)
        
    plt.xlabel("Pathologist Grade (Steatosis)")
    plt.ylabel("AI Pathological Fat Ratio")
    plt.title("Fig 6-3: AI Fat Ratio Validation")
    save_fig("Fig_6_3_AI_Pathology_Validation")

def main():
    ensure_dir(OUTPUT_DIR)
    set_style()
    df = load_master()
    
    # SHAP
    shap_vals_f, X_res_f, feats_f = fig_5_1_2_shap(df)
    
    # Dependence (using F_Cirr data)
    fig_5_3_dependence(shap_vals_f, X_res_f, feats_f)
    
    # Human vs AI
    fig_5_4_human_vs_ai()
    
    # Pathology
    fig_6_2_3_pathology_corr(df)
    
    logging.info("All Validation Figures Generated.")

if __name__ == "__main__":
    main()
