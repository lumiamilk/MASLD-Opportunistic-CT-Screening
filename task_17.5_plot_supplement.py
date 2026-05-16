import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from PIL import Image, ImageDraw, ImageFont
import logging

# Set PIL max image size to None to handle large pathology images
Image.MAX_IMAGE_PIXELS = None

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
MASTER_CSV = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "thesis_assets", "figures")
ASSETS_V2 = os.path.join(BASE_DIR, "output", "figure_assets_hd_v2")
ASSETS_V5 = os.path.join(BASE_DIR, "output", "figure_assets_hd_v5.3")
RETICULIN_IMG = os.path.join(BASE_DIR, "output", "reticulin_case_study.png")

# Patient ID for Case Study (from Task 14 Report)
# A case where Human Missed (Reported Normal) but AI Found (S3)
CASE_ID_S3 = "2020027543" 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['figure.dpi'] = 300

def load_data():
    try: df = pd.read_csv(MASTER_CSV, encoding='utf-8-sig')
    except: df = pd.read_csv(MASTER_CSV, encoding='gb18030')
    mask = df['L_S_Ratio'].notna() | df['original_firstorder_Energy'].notna()
    return df[mask].copy()

# --- Fig 5-5: Radar Chart ---
def normalize_column(df, col, invert=False):
    """Normalize column to 0-1 range. If invert is True, Low=1, High=0 (Badness score)."""
    min_val = df[col].min()
    max_val = df[col].max()
    
    if max_val == min_val:
        return np.zeros(len(df))
    
    norm = (df[col] - min_val) / (max_val - min_val)
    if invert:
        return 1.0 - norm
    else:
        return norm

def fig_5_5_radar(df):
    logging.info("Generating Fig 5-5: Radar Chart")
    
    # 1. Select Cases
    # Case S3: Specific ID
    case_s3 = df[df['patient_id'].astype(str) == CASE_ID_S3]
    if case_s3.empty:
        logging.warning(f"Case {CASE_ID_S3} not found in dataframe. Picking random S3 case.")
        case_s3 = df[df['Steatosis_Grade'] == 3].sample(1, random_state=42)
    
    # Case S0/S1: Low Risk Control (Fusion cohort has no S0, so we pick S1F0)
    control_mask = (df['Steatosis_Grade'] == 1) & (df['Fibrosis_Stage'] == 0)
    if control_mask.any():
        case_s0 = df[control_mask].sample(1, random_state=42)
        control_label = 'Low Risk Control (S1F0)'
    else:
        # Fallback to just S1 if F0 not found
        case_s0 = df[df['Steatosis_Grade'] == 1].sample(1, random_state=42)
        control_label = 'Low Risk Control (S1)'
    
    # 2. Prepare Features
    # Features: Liver_HU (inv), L_S_Ratio (inv), Texture_Energy (inv), FIB-4, BMI
    cols_map = {
        'Liver_Mean_HU': ('Liver Density', True),
        'L_S_Ratio': ('L/S Ratio', True),
        'original_firstorder_Energy': ('Texture Uniformity', True),
        'Index_FIB4': ('FIB-4 Index', False),
        'BMI': ('BMI', False)
    }
    
    # Create normalized dataframe for the WHOLE population to establish scale
    df_norm = df.copy()
    for col, (label, invert) in cols_map.items():
        if col not in df.columns:
            logging.warning(f"Column {col} missing.")
            continue
        df_norm[col + '_norm'] = normalize_column(df, col, invert)
        
    # Get values for plotting
    # We want to plot the Normalized values (0-1), representing "Risk/Severity"
    
    categories = [cols_map[c][0] for c in cols_map.keys()]
    N = len(categories)
    
    # Values
    values_s3 = []
    values_s0 = []
    
    for col in cols_map.keys():
        if col + '_norm' in df_norm.columns:
            # Get value for the specific row using index
            v_s3 = df_norm.loc[case_s3.index[0], col + '_norm']
            v_s0 = df_norm.loc[case_s0.index[0], col + '_norm']
            values_s3.append(v_s3)
            values_s0.append(v_s0)
        else:
            values_s3.append(0)
            values_s0.append(0)
            
    # Close the loop
    values_s3 += values_s3[:1]
    values_s0 += values_s0[:1]
    
    # Angles
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + labels
    plt.xticks(angles[:-1], categories)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot S3
    ax.plot(angles, values_s3, linewidth=2, linestyle='solid', label='AI Detected S3 (Human Missed)', color='red')
    ax.fill(angles, values_s3, 'red', alpha=0.1)
    
    # Plot S0
    ax.plot(angles, values_s0, linewidth=2, linestyle='solid', label=control_label, color='blue')
    ax.fill(angles, values_s0, 'blue', alpha=0.1)
    
    plt.title("Fig 5-5: Multimodal Risk Profile (Radar Chart)", y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Save
    path_png = os.path.join(OUTPUT_DIR, "Fig_5_5_Radar_Case_Study.png")
    path_tif = os.path.join(OUTPUT_DIR, "Fig_5_5_Radar_Case_Study.tif")
    plt.savefig(path_png, bbox_inches='tight', dpi=300)
    plt.savefig(path_tif, bbox_inches='tight', dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    logging.info("Saved Fig 5-5")
    plt.close()

# --- Fig 6-1: Multimodal Panel ---
def fig_6_1_panel():
    logging.info("Generating Fig 6-1: Multimodal Panel (2x2 Layout)")
    
    # Find images for Case 2020027543
    # A: CT
    p_ct = os.path.join(ASSETS_V2, f"CT_{CASE_ID_S3}.png")
    
    # B: HE Original (ROI)
    p_he = os.path.join(ASSETS_V2, f"Patho_{CASE_ID_S3}_ROI.jpg")
    if not os.path.exists(p_he):
         p_he = os.path.join(ASSETS_V5, f"{CASE_ID_S3}_ROI_V4.jpg")
         
    # C: HE Mask (Fat Seg)
    p_mask = os.path.join(ASSETS_V2, f"Patho_{CASE_ID_S3}_Mask.jpg")
    
    # D: Reticulin
    p_ret = RETICULIN_IMG
    
    # List of files
    files = [p_ct, p_he, p_mask, p_ret]
    labels = ["A. CT Slice", "B. H&E Original", "C. AI Fat Quantification", "D. Reticulin (Fibrosis)"]
    
    target_size = (1024, 1024)
    images = []
    
    for f in files:
        if os.path.exists(f):
            try:
                img = Image.open(f)
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                images.append(img)
            except Exception as e:
                logging.error(f"Failed to load {f}: {e}")
                images.append(None)
        else:
            logging.warning(f"File not found: {f}")
            images.append(None)
            
    # Create blank if missing
    for i in range(len(images)):
        if images[i] is None:
            images[i] = Image.new('RGB', target_size, color='lightgrey')
            
    # Stitch 2x2
    # Width = 2*W + padding, Height = 2*H + padding
    padding = 20
    panel_w = 2 * target_size[0] + 3 * padding
    panel_h = 2 * target_size[1] + 3 * padding
    
    panel = Image.new('RGB', (panel_w, panel_h), color='white')
    draw = ImageDraw.Draw(panel)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 60)
    except:
        font = ImageFont.load_default()
    
    # Positions: (0,0), (1,0), (0,1), (1,1)
    grid_pos = [(0, 0), (1, 0), (0, 1), (1, 1)]
    
    for i, (col, row) in enumerate(grid_pos):
        img = images[i]
        x = padding + col * (target_size[0] + padding)
        y = padding + row * (target_size[1] + padding)
        
        panel.paste(img, (x, y))
        
        # Add Label Overlay
        label_text = labels[i]
        # Draw text with outline for visibility
        text_pos = (x + 20, y + 20)
        draw.text(text_pos, label_text, fill="white", stroke_width=3, stroke_fill="black", font=font)
        
    # Save
    path_png = os.path.join(OUTPUT_DIR, "Fig_6_1_Multimodal_Panel.png")
    path_tif = os.path.join(OUTPUT_DIR, "Fig_6_1_Multimodal_Panel.tif")
    panel.save(path_png)
    panel.save(path_tif, compression="tiff_lzw")
    logging.info("Saved Fig 6-1 (2x2)")

def main():
    ensure_dir(OUTPUT_DIR)
    set_style()
    df = load_data()
    
    fig_5_5_radar(df)
    fig_6_1_panel()
    
    logging.info("All Supplement Figures Generated.")

if __name__ == "__main__":
    main()
