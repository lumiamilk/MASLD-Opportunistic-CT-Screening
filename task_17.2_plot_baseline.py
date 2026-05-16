import pandas as pd
import numpy as np
import os
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib import font_manager
import logging

# --- Configuration ---
BASE_DIR = os.getcwd()
MASTER_CSV = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
DB_PATH = os.path.join(BASE_DIR, "output", "task_1_ct_metadata.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "thesis_assets", "figures")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_style():
    # Try to find Chinese fonts
    fonts = [f.name for f in font_manager.fontManager.ttflist]
    target_fonts = ['SimHei', 'SimSun', 'Microsoft YaHei', 'Arial Unicode MS']
    selected_font = 'DejaVu Sans' # Fallback
    
    for font in target_fonts:
        if any(f.lower() == font.lower() for f in fonts): # Exact match check might be tricky with system names
            # Let's try to find it in the list
            found = [f for f in fonts if font.lower() in f.lower()]
            if found:
                selected_font = found[0]
                break
    
    # In WSL, sometimes fonts aren't installed. We might fallback to English.
    # We will just set a default sans-serif and hope for the best, or prioritize Arial.
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False # Fix minus sign
    plt.rcParams['figure.dpi'] = 300
    
    logging.info(f"Plotting style set. Font: {selected_font}")

def load_data():
    try: df = pd.read_csv(MASTER_CSV, encoding='utf-8-sig')
    except: df = pd.read_csv(MASTER_CSV, encoding='gb18030')
    
    # Preprocess strings to numeric if needed, though they should be fine from master csv
    return df

def get_fusion_cohort(df):
    mask = df['L_S_Ratio'].notna() | df['original_firstorder_Energy'].notna()
    return df[mask].copy()

def save_fig(name):
    path_png = os.path.join(OUTPUT_DIR, f"{name}.png")
    path_tif = os.path.join(OUTPUT_DIR, f"{name}.tif")
    plt.savefig(path_png, bbox_inches='tight', dpi=300)
    plt.savefig(path_tif, bbox_inches='tight', dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    logging.info(f"Saved {name}")
    plt.close()

# --- Task 1: Consort Diagram Numbers ---
def fig_3_1_consort(df):
    total = len(df)
    fusion = get_fusion_cohort(df)
    included = len(fusion)
    excluded = total - included
    
    # Try to guess exclusion reasons if possible (e.g., missing CT data)
    # In master csv, we don't have a specific "reason" column, but we know absence of L_S_Ratio means CT not processed.
    
    lines = [
        "FIG 3-1 CONSORT FLOW DIAGRAM DATA",
        "=================================",
        f"1. Total Patients Screened (Biopsy): {total}",
        f"2. Excluded (No CT / Quality Fail): {excluded}",
        f"3. Final Included Cohort (Fusion): {included}",
        "   (Use these numbers to draw the flowchart)"
    ]
    
    with open(os.path.join(OUTPUT_DIR, "consort_numbers.txt"), "w") as f:
        f.write("\n".join(lines))
    logging.info("Generated consort_numbers.txt")

# --- Task 2: Data Quality ---
def fig_3_2_quality():
    if not os.path.exists(DB_PATH):
        logging.warning("DB not found, skipping Fig 3-2")
        return

    conn = sqlite3.connect(DB_PATH)
    df_meta = pd.read_sql("SELECT slice_thickness, kernel, kvp FROM ct_series", conn)
    conn.close()
    
    # Filter reasonable values (ignore outliers/test data if any)
    df_meta = df_meta[df_meta['slice_thickness'] > 0]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # A: Slice Thickness
    sns.histplot(data=df_meta, x='slice_thickness', kde=False, ax=axes[0], bins=10, color='skyblue')
    axes[0].set_title('A. 层厚分布', loc='left', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('层厚 (mm)', fontsize=12)
    axes[0].set_ylabel('频数', fontsize=12)
    
    # B: Kernel
    # Top 5 kernels
    top_kernels = df_meta['kernel'].value_counts().nlargest(5).index
    df_kernel = df_meta[df_meta['kernel'].isin(top_kernels)]
    sns.countplot(data=df_kernel, x='kernel', ax=axes[1], order=top_kernels, palette='viridis')
    axes[1].set_title('B. 重建核类型分布 (Top 5)', loc='left', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('重建核类型', fontsize=12)
    axes[1].set_ylabel('频数', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    
    # C: KVP
    sns.countplot(data=df_meta, x='kvp', ax=axes[2], palette='Set2')
    axes[2].set_title('C. 管电压分布', loc='left', fontweight='bold', fontsize=14)
    axes[2].set_xlabel('管电压 (kV)', fontsize=12)
    axes[2].set_ylabel('频数', fontsize=12)
    
    plt.tight_layout()
    save_fig("Fig_3_2_Data_Quality")

# --- Helper: P-value Annotation ---
def add_p_value_annotation(ax, df, x, y, test_method='kruskal'):
    """
    Adds p-value to the plot.
    """
    # Group data
    groups = []
    labels = sorted(df[x].unique())
    for label in labels:
        groups.append(df[df[x] == label][y].dropna().values)
    
    if len(groups) < 2:
        return
        
    if test_method == 'kruskal':
        stat, p = stats.kruskal(*groups)
    else:
        stat, p = stats.f_oneway(*groups) # ANOVA
        
    # Format p-value
    p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    
    # Add text to top right
    ax.text(0.95, 0.95, p_text, transform=ax.transAxes, 
            ha='right', va='top', fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

# --- Task 3: Clinical vs Pathology ---
def fig_3_3_clinical_violin(df):
    fusion = get_fusion_cohort(df)
    
    # Features to plot
    # Rows: 2, Cols: 3
    features = ['ALT_Val', 'AST_Val', 'Index_FIB4', 'Index_APRI', 'PLT_Val', 'TG_Val']
    titles = ['ALT', 'AST', 'FIB-4', 'APRI', 'Platelets', 'Triglycerides']
    
    # We use Steatosis for TG, Fibrosis for others?
    # Prompt says: X轴: Steatosis_Grade (0-3) 或 Fibrosis_Stage (0-4).
    # Usually FIB4/APRI/PLT relate to Fibrosis. ALT/AST/TG relate to Steatosis/Inflammation.
    # Let's split:
    # Row 1: Steatosis -> ALT, AST, TG
    # Row 2: Fibrosis -> FIB-4, APRI, PLT
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Row 1: vs Steatosis
    row1_feats = ['ALT_Val', 'AST_Val', 'TG_Val']
    row1_titles = ['ALT (U/L)', 'AST (U/L)', 'Triglycerides (mmol/L)']
    
    for i, feat in enumerate(row1_feats):
        ax = axes[i]
        sns.violinplot(data=fusion, x='Steatosis_Grade', y=feat, ax=ax, palette='Blues', inner='quartile')
        ax.set_title(f"{row1_titles[i]} by Steatosis")
        ax.set_xlabel("Steatosis Grade")
        ax.set_ylabel(row1_titles[i])
        add_p_value_annotation(ax, fusion, 'Steatosis_Grade', feat)

    # Row 2: vs Fibrosis
    row2_feats = ['Index_FIB4', 'Index_APRI', 'PLT_Val']
    row2_titles = ['FIB-4 Index', 'APRI Index', 'Platelets (10^9/L)']
    
    for i, feat in enumerate(row2_feats):
        ax = axes[i+3]
        sns.violinplot(data=fusion, x='Fibrosis_Stage', y=feat, ax=ax, palette='Oranges', inner='quartile')
        ax.set_title(f"{row2_titles[i]} by Fibrosis")
        ax.set_xlabel("Fibrosis Stage")
        ax.set_ylabel(row2_titles[i])
        # Handle outliers for plot scaling if needed (e.g. FIB4 can be huge)
        # Clip Y for visualization if extreme outliers? Let's leave for now.
        add_p_value_annotation(ax, fusion, 'Fibrosis_Stage', feat)
        
    plt.tight_layout()
    save_fig("Fig_3_3_Clinical_vs_Pathology")

# --- Task 4: CT Features vs Pathology ---
def fig_3_4_ct_features_box(df):
    fusion = get_fusion_cohort(df)
    
    # 2 Rows, 3 Cols
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Row 1: Steatosis vs CT Features
    # Liver_Mean_HU, L_S_Ratio, Fat_Volume
    s_feats = ['Liver_Mean_HU', 'L_S_Ratio', 'Fat_Volume']
    s_labels = ['Liver CT Value (HU)', 'L/S Ratio', 'Fat Volume (cm³)']
    
    for i, feat in enumerate(s_feats):
        ax = axes[i]
        sns.boxplot(data=fusion, x='Steatosis_Grade', y=feat, ax=ax, palette='viridis', showfliers=False)
        sns.stripplot(data=fusion, x='Steatosis_Grade', y=feat, ax=ax, color='black', alpha=0.3, size=3)
        ax.set_title(f"{s_labels[i]} vs Steatosis")
        ax.set_xlabel("Steatosis Grade")
        add_p_value_annotation(ax, fusion, 'Steatosis_Grade', feat)
        
    # Row 2: Fibrosis vs CT Features
    # Spleen_Volume, Muscle_Mean_HU, VAT_SAT_Ratio
    f_feats = ['Spleen_Volume', 'Muscle_Mean_HU', 'VAT_SAT_Ratio']
    f_labels = ['Spleen Volume (cm³)', 'Muscle Density (HU)', 'VAT/SAT Ratio']
    
    for i, feat in enumerate(f_feats):
        ax = axes[i+3]
        sns.boxplot(data=fusion, x='Fibrosis_Stage', y=feat, ax=ax, palette='magma', showfliers=False)
        sns.stripplot(data=fusion, x='Fibrosis_Stage', y=feat, ax=ax, color='black', alpha=0.3, size=3)
        ax.set_title(f"{f_labels[i]} vs Fibrosis")
        ax.set_xlabel("Fibrosis Stage")
        add_p_value_annotation(ax, fusion, 'Fibrosis_Stage', feat)

    plt.tight_layout()
    save_fig("Fig_3_4_CT_vs_Pathology")

# --- Task 5: Correlation Matrix ---
def fig_3_5_correlation(df):
    fusion = get_fusion_cohort(df)
    
    # Select Core Features
    clinical = ['age', 'BMI', 'ALT_Val', 'AST_Val', 'PLT_Val', 'TG_Val', 'Index_FIB4', 'Index_APRI']
    pathology = ['Steatosis_Grade', 'Fibrosis_Stage', 'Inflammation_Grade']
    ct_quant = ['Liver_Mean_HU', 'L_S_Ratio', 'Fat_Volume', 'Visceral_Fat_Volume', 'Spleen_Volume', 'VAT_SAT_Ratio']
    
    # Select Top Radiomics (Variance based or pre-selected)
    # We'll just grab a few representatives if Variance ranking isn't handy here,
    # or grab 'original_firstorder_Energy' etc.
    # Let's grab specific ones often important
    radiomics = [
        'original_firstorder_Mean', 'original_firstorder_Skewness',
        'original_glcm_Contrast', 'original_glcm_Correlation',
        'original_glrlm_RunEntropy', 'original_glszm_ZonePercentage'
    ]
    
    selected_cols = pathology + clinical + ct_quant + radiomics
    # Filter only existing columns
    selected_cols = [c for c in selected_cols if c in fusion.columns]
    
    # Rename for pretty plotting
    nice_names = {
        'Steatosis_Grade': 'Steatosis', 'Fibrosis_Stage': 'Fibrosis', 'Inflammation_Grade': 'Inflammation',
        'ALT_Val': 'ALT', 'AST_Val': 'AST', 'PLT_Val': 'PLT', 'TG_Val': 'TG',
        'Index_FIB4': 'FIB-4', 'Index_APRI': 'APRI',
        'Liver_Mean_HU': 'Liver HU', 'L_S_Ratio': 'L/S Ratio',
        'Fat_Volume': 'Fat Vol', 'Visceral_Fat_Volume': 'VAT Vol', 'Spleen_Volume': 'Spleen Vol',
        'VAT_SAT_Ratio': 'VAT/SAT',
        'original_firstorder_Mean': 'Rad: Mean', 'original_firstorder_Skewness': 'Rad: Skew',
        'original_glcm_Contrast': 'Rad: Contrast', 'original_glcm_Correlation': 'Rad: Corr',
        'original_glrlm_RunEntropy': 'Rad: RunEntropy', 'original_glszm_ZonePercentage': 'Rad: Zone%' 
    }
    
    data = fusion[selected_cols].rename(columns=nice_names)
    
    # Calculate Correlation
    corr = data.corr(method='spearman')
    
    # Calculate P-values
    p_values = np.zeros_like(corr)
    for i, col1 in enumerate(data.columns):
        for j, col2 in enumerate(data.columns):
            if i == j: 
                p_values[i, j] = 1.0
            else:
                # dropna for pair
                pair = data[[col1, col2]].dropna()
                if len(pair) > 2:
                    _, p = stats.spearmanr(pair[col1], pair[col2])
                    p_values[i, j] = p
                else:
                    p_values[i, j] = 1.0
                    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Plot
    plt.figure(figsize=(16, 14))
    ax = sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
                     vmax=1, vmin=-1, square=True, linewidths=.5,
                     cbar_kws={"shrink": .5})
    
    # Add stars for significance
    # Iterate over data to add text
    for i in range(len(corr)):
        for j in range(len(corr)):
            if i > j: # Lower triangle only
                p = p_values[i, j]
                if p < 0.05:
                    text = '*' if p < 0.01 else '.'
                    # Only mark strong ones to avoid clutter? Or standard * <0.05, ** <0.01
                    text = '**' if p < 0.01 else '*'
                    ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', color='black', fontsize=10)
    
    plt.title("Spearman Correlation Matrix (Significance: * p<0.05, ** p<0.01)")
    save_fig("Fig_3_5_Correlation_Matrix")

def main():
    ensure_dir(OUTPUT_DIR)
    set_style()
    
    df = load_data()
    
    fig_3_1_consort(df)
    fig_3_2_quality()
    fig_3_3_clinical_violin(df)
    fig_3_4_ct_features_box(df)
    fig_3_5_correlation(df)
    
    logging.info("All Baseline Figures Generated.")

if __name__ == "__main__":
    main()
