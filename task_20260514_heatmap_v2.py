import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# --- Chinese translation dict (from task_18_plot_CN.py) ---
TRANS_DICT = {
    "Target": "预测靶点",
    "Model": "模型",
    "AUC": "AUC",
    "Machine Learning Model": "机器学习模型",
    "Clinical Target": "临床预测靶点",
    "AUC (Mean)": "AUC (均值)",
    "S_Mild (S>=1)": "轻度脂肪变性 (S≥1)",
    "S_Mod (S>=2)": "中重度脂肪变性 (S≥2)",
    "S_Sev (S=3)": "重度脂肪变性 (S=3)",
    "F_Adv (F>=3)": "进展期纤维化 (F≥3)",
    "F_Cirr (F=4)": "肝硬化 (F=4)",
    "LR": "逻辑回归",
    "SVM": "支持向量机",
    "RF": "随机森林",
    "XGB": "XGBoost",
    "LGBM": "LightGBM",
    "CAT": "CatBoost",
    "MLP": "多层感知机",
}

def translate(text):
    if not isinstance(text, str):
        return text
    if text in TRANS_DICT:
        return TRANS_DICT[text]
    for k, v in TRANS_DICT.items():
        if k in text and len(k) > 3:
            return text.replace(k, v)
    return text

# --- Paths ---
TABLE_4_2_CSV = r"D:\mWork\paper0\output\thesis_assets\tables\task_17.1_table_4_2_multi_target.csv"
OUTPUT_DIR = r"D:\mWork\paper0\output\20260514"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Font setup ---
font_path = r"C:\Windows\Fonts\simhei.ttf"
if os.path.exists(font_path):
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = [prop.get_name()] + plt.rcParams['font.sans-serif']
else:
    print(f"WARNING: SimHei font not found at {font_path}")

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
if os.path.exists(font_path):
    plt.rcParams['font.sans-serif'] = [prop.get_name()] + plt.rcParams['font.sans-serif']

# --- Read data ---
df = pd.read_csv(TABLE_4_2_CSV)
print("Original data:")
print(df)

# --- DROP rows where Target contains "S>=1" (i.e., S_Mild) ---
mask_drop = df['Target'].str.contains(r'S>=1', na=False)
print(f"\nDropping rows: {df[mask_drop]['Target'].tolist()}")
df = df[~mask_drop].copy()
print(f"\nAfter dropping (S>=1):")
print(df)

df_plot = df.set_index('Target')

def extract_mean(val):
    if isinstance(val, str) and '±' in val:
        return float(val.split('±')[0].strip())
    try:
        return float(val)
    except:
        return np.nan

df_numeric = df_plot.applymap(extract_mean)

# Translate row and column labels to Chinese
df_numeric.index = [translate(i) for i in df_numeric.index]
df_numeric.columns = [translate(c) for c in df_numeric.columns]

print("\nFinal data for heatmap:")
print(df_numeric)

# --- Plot ---
plt.figure(figsize=(10, 5.5))
sns.heatmap(
    df_numeric,
    annot=True,
    cmap="RdYlGn",
    fmt=".3f",
    vmin=0.5,
    vmax=0.95,
    cbar_kws={'label': TRANS_DICT['AUC (Mean)']}
)
plt.xlabel(TRANS_DICT['Machine Learning Model'])
plt.ylabel(TRANS_DICT['Clinical Target'])

output_path = os.path.join(OUTPUT_DIR, "Fig_4_1_Multi_Target_Heatmap.png")
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"\nSaved: {output_path}")
