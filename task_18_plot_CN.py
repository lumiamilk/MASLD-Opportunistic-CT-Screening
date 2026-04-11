import pandas as pd
import numpy as np
import os
import sqlite3
import re
import matplotlib
matplotlib.use('Agg') # Ensure non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy import stats
from matplotlib import font_manager
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibrationDisplay
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from PIL import Image, ImageDraw, ImageFont

# --- Configuration & Paths ---
BASE_DIR = "/mnt/d/mWork/paper0"
MASTER_CSV = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
DB_PATH = os.path.join(BASE_DIR, "output", "task_1_ct_metadata.db")
QUANT_RESULTS_CSV = os.path.join(BASE_DIR, "output", "task_16_quant_results_v5.3.csv")
TABLE_4_2_CSV = os.path.join(BASE_DIR, "output", "thesis_assets", "tables", "task_17.1_table_4_2_multi_target.csv")
TABLE_5_2_CSV = os.path.join(BASE_DIR, "output", "thesis_assets", "tables", "task_17.1_table_5_2_human_vs_ai.csv")
IMG_ASSETS_DIR = os.path.join(BASE_DIR, "output")

# Input/Output Dirs
INPUT_DIR_TABLES = os.path.join(BASE_DIR, "output", "thesis_assets", "tables")
OUTPUT_DIR_TABLES = os.path.join(BASE_DIR, "output", "thesis_assets", "tables_cn")
OUTPUT_DIR_FIGURES = os.path.join(BASE_DIR, "output", "thesis_assets", "figures_cn")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Localization Dictionary ---
TRANS_DICT = {
    # --- Table Headers & Values ---
    "Target": "预测靶点",
    "Model": "模型",
    "AUC": "AUC",
    "Accuracy": "准确率",
    "Sensitivity": "敏感度",
    "Specificity": "特异度",
    "PPV": "阳性预测值",
    "NPV": "阴性预测值",
    "F1 Score": "F1分数",
    "Support": "样本量",
    "Variable": "变量",
    "Feature": "特征",
    "Value": "数值",
    "P_Value": "P值",
    "Method": "诊断方法",
    "CI_Lower": "95% CI 下限",
    "CI_Upper": "95% CI 上限",
    "Threshold": "截断值",
    "Rank": "排名",
    "Mean_SHAP": "平均SHAP值",
    
    # Models
    "LogisticRegression": "逻辑回归",
    "SVM": "支持向量机",
    "RandomForest": "随机森林",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "Clinical": "临床模型",
    "Radiomics": "影像组学模型",
    "BodyComp": "体成分模型",
    "Fusion": "融合模型",
    
    # Targets / Groups
    "Steatosis_Grade": "脂肪变性分级",
    "Fibrosis_Stage": "纤维化分期",
    "Inflammation_Grade": "炎症分级",
    "S_Sev": "重度脂肪变性 (S≥3)",
    "F_Cirr": "肝硬化 (F=4)",
    "F_Sig": "显著纤维化 (F≥2)",
    "Normal": "正常",
    "Steatosis": "脂肪肝",
    "Fibrosis": "纤维化",
    
    # --- Features (Clinical) ---
    "Age": "年龄",
    "age": "年龄",
    "Sex": "性别",
    "sex": "性别",
    "Male Sex": "男性",
    "BMI": "体质量指数 (BMI)",
    "T2DM": "2型糖尿病",
    "High_Blood_pressure": "高血压",
    "ALT_Val": "谷丙转氨酶 (ALT)",
    "AST_Val": "谷草转氨酶 (AST)",
    "PLT_Val": "血小板计数 (PLT)",
    "TG_Val": "甘油三酯 (TG)",
    "Index_FIB4": "FIB-4指数",
    "Index_APRI": "APRI指数",
    
    # --- Features (CT/Body) ---
    "Liver_Mean_HU": "肝脏CT值 (HU)",
    "L_S_Ratio": "肝/脾CT值比率",
    "Muscle_Mean_HU": "骨骼肌密度 (HU)",
    "Fat_Volume": "皮下脂肪体积 (SAT)",
    "Visceral_Fat_Volume": "内脏脂肪体积 (VAT)",
    "Spleen_Volume": "脾脏体积 (cm³)",
    "VAT_SAT_Ratio": "内脏/皮下脂肪比",
    
    # --- Features (Radiomics - Simplify names) ---
    "original_firstorder_Energy": "能量 (Energy)",
    "original_firstorder_Mean": "均值 (Mean)",
    "original_firstorder_Skewness": "偏度 (Skewness)",
    "original_glcm_Contrast": "对比度 (Contrast)",
    "original_glcm_Correlation": "相关性 (Correlation)",
    "original_glrlm_RunEntropy": "游程熵 (RunEntropy)",
    "original_glszm_ZonePercentage": "区域百分比 (Zone%)",
    
    # --- Plot Labels ---
    "True Positive Rate": "敏感度 (Sensitivity)",
    "False Positive Rate": "1 - 特异度 (1 - Specificity)",
    "Net Benefit": "净获益 (Net Benefit)",
    "Threshold Probability": "阈值概率",
    "Treat All": "全部治疗",
    "Treat None": "不治疗",
    "Fraction of positives": "阳性比例 (Observed)",
    "Mean predicted probability": "预测概率 (Predicted)",
    "SHAP value": "SHAP值 (对模型输出的影响)",
    "Feature value": "特征值",
    "Low": "低",
    "High": "高",
    "Human": "病理医生",
    "AI": "人工智能",
    "AI Pathological Fat Ratio": "AI 病理脂肪比例",
    "Pathologist Grade": "病理医生分级",
    "Liver CT Value (HU)": "肝脏CT值 (HU)",
    "Slice Thickness": "层厚 (mm)",
    "Kernel Type": "卷积核类型",
    "Tube Voltage": "管电压 (kV)",
    "Count": "频数",
    "Correlation": "相关系数",
    "Sensitivity (Recall)": "敏感度 (Recall)",
    "Machine Learning Model": "机器学习模型",
    "Clinical Target": "临床预测靶点",
    "AUC (Mean)": "AUC (均值)",
    "Threshold": "阈值",
    "Net Benefit": "净收益"
}

# --- Utils ---
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_style():
    # Explicitly load Chinese font
    font_path = '/mnt/c/Windows/Fonts/simhei.ttf'
    if os.path.exists(font_path):
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.sans-serif'] = [prop.get_name()] + plt.rcParams['font.sans-serif']
    else:
        logging.warning(f"SimHei font not found at {font_path}")
        
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4) # +0.2 scale
    # Re-apply font
    if os.path.exists(font_path):
         plt.rcParams['font.sans-serif'] = [prop.get_name()] + plt.rcParams['font.sans-serif']

def load_data():
    try: df = pd.read_csv(MASTER_CSV, encoding='utf-8-sig')
    except: df = pd.read_csv(MASTER_CSV, encoding='gb18030')
    return df

def get_fusion_cohort(df):
    mask = df['L_S_Ratio'].notna() | df['original_firstorder_Energy'].notna()
    return df[mask].copy()

def save_fig(name):
    if not name.endswith('.png'): name += '.png'
    path_png = os.path.join(OUTPUT_DIR_FIGURES, name)
    plt.savefig(path_png, bbox_inches='tight', dpi=300)
    logging.info(f"Saved {name}")
    plt.close()

def translate(text):
    if not isinstance(text, str): return text
    if text in TRANS_DICT: return TRANS_DICT[text]
    for k, v in TRANS_DICT.items():
        if k in text and len(k) > 3: # Avoid short matches
             return text.replace(k, v)
    return text

# ==========================================
# MODULE 1: BASELINE (from task_17.2)
# ==========================================

def fig_3_2_quality():
    if not os.path.exists(DB_PATH): return
    conn = sqlite3.connect(DB_PATH)
    df_meta = pd.read_sql("SELECT slice_thickness, kernel, kvp FROM ct_series", conn)
    conn.close()
    df_meta = df_meta[df_meta['slice_thickness'] > 0]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.histplot(data=df_meta, x='slice_thickness', kde=False, ax=axes[0], bins=10, color='skyblue')
    # axes[0].set_title('') # REMOVED
    axes[0].set_xlabel(TRANS_DICT['Slice Thickness'])
    axes[0].set_ylabel(TRANS_DICT['Count'])
    
    top_kernels = df_meta['kernel'].value_counts().nlargest(5).index
    df_kernel = df_meta[df_meta['kernel'].isin(top_kernels)]
    sns.countplot(data=df_kernel, x='kernel', ax=axes[1], order=top_kernels, palette='viridis')
    # axes[1].set_title('') # REMOVED
    axes[1].set_xlabel(TRANS_DICT['Kernel Type'])
    axes[1].set_ylabel(TRANS_DICT['Count'])
    axes[1].tick_params(axis='x', rotation=45)
    
    sns.countplot(data=df_meta, x='kvp', ax=axes[2], palette='Set2')
    # axes[2].set_title('') # REMOVED
    axes[2].set_xlabel(TRANS_DICT['Tube Voltage'])
    axes[2].set_ylabel(TRANS_DICT['Count'])
    
    plt.tight_layout()
    save_fig("Fig_3_2_Data_Quality")

def add_p_value_annotation(ax, df, x, y):
    groups = [df[df[x] == label][y].dropna().values for label in sorted(df[x].unique())]
    if len(groups) < 2: return
    stat, p = stats.kruskal(*groups)
    p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.95, 0.95, p_text, transform=ax.transAxes, ha='right', va='top', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

def fig_3_3_clinical_violin(df):
    fusion = get_fusion_cohort(df)
    features = ['ALT_Val', 'AST_Val', 'TG_Val', 'Index_FIB4', 'Index_APRI', 'PLT_Val']
    x_cols = ['Steatosis_Grade']*3 + ['Fibrosis_Stage']*3 # Matches 17.2 logic
    titles = [TRANS_DICT[f] for f in features] # Mapped titles as Y labels
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        x_col = x_cols[i]
        ax = axes[i]
        sns.violinplot(data=fusion, x=x_col, y=feat, ax=ax, palette='Blues' if i<3 else 'Oranges', inner='quartile')
        # ax.set_title('') # REMOVED
        ax.set_xlabel(TRANS_DICT[x_col])
        ax.set_ylabel(titles[i])
        add_p_value_annotation(ax, fusion, x_col, feat)
        
    plt.tight_layout()
    save_fig("Fig_3_3_Clinical_vs_Pathology")

def fig_3_4_ct_features_box(df):
    fusion = get_fusion_cohort(df)
    features = ['Liver_Mean_HU', 'L_S_Ratio', 'Fat_Volume', 'Spleen_Volume', 'Muscle_Mean_HU', 'VAT_SAT_Ratio']
    x_cols = ['Steatosis_Grade']*3 + ['Fibrosis_Stage']*3
    titles = [TRANS_DICT[f] for f in features]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        x_col = x_cols[i]
        ax = axes[i]
        sns.boxplot(data=fusion, x=x_col, y=feat, ax=ax, palette='viridis' if i<3 else 'magma', showfliers=False)
        sns.stripplot(data=fusion, x=x_col, y=feat, ax=ax, color='black', alpha=0.3, size=3)
        # ax.set_title('') # REMOVED
        ax.set_xlabel(TRANS_DICT[x_col])
        ax.set_ylabel(titles[i])
        add_p_value_annotation(ax, fusion, x_col, feat)

    plt.tight_layout()
    save_fig("Fig_3_4_CT_vs_Pathology")

def fig_3_5_correlation(df):
    fusion = get_fusion_cohort(df)
    cols = ['Steatosis_Grade', 'Fibrosis_Stage', 'ALT_Val', 'AST_Val', 'Index_FIB4', 
            'Liver_Mean_HU', 'L_S_Ratio', 'Fat_Volume', 'Spleen_Volume']
    # Simplified labels for matrix
    names = [TRANS_DICT.get(c, c).replace(" (HU)", "").replace(" (cm³)", "") for c in cols]
    
    data = fusion[cols].copy()
    data.columns = names
    corr = data.corr(method='spearman')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, vmax=1, vmin=-1, 
                annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .5})
    # plt.title('') # REMOVED
    save_fig("Fig_3_5_Correlation_Matrix")


# ==========================================
# MODULE 2: MODELING (from task_17.3)
# ==========================================

def fig_4_1_heatmap():
    if not os.path.exists(TABLE_4_2_CSV): return
    df = pd.read_csv(TABLE_4_2_CSV)
    df_plot = df.set_index('Target')
    
    def extract_mean(val):
        if isinstance(val, str) and '±' in val: return float(val.split('±')[0].strip())
        try: return float(val)
        except: return np.nan
            
    df_numeric = df_plot.applymap(extract_mean)
    # Translate
    df_numeric.index = [translate(i) for i in df_numeric.index]
    df_numeric.columns = [translate(c) for c in df_numeric.columns]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_numeric, annot=True, cmap="RdYlGn", fmt=".3f", 
                vmin=0.5, vmax=0.95, cbar_kws={'label': TRANS_DICT['AUC (Mean)']})
    # plt.title('')
    plt.xlabel(TRANS_DICT['Machine Learning Model'])
    plt.ylabel(TRANS_DICT['Clinical Target'])
    save_fig("Fig_4_1_Multi_Target_Heatmap")

def get_feature_sets(df):
    rad_cols = [c for c in df.columns if c.startswith('original_')]
    clinical_cols = ['age', 'sex', 'BMI', 'T2DM', 'High_Blood_pressure', 'ALT_Val', 'AST_Val', 'PLT_Val', 'TG_Val']
    body_cols = ['Liver_Mean_HU', 'L_S_Ratio', 'Muscle_Mean_HU', 'Fat_Volume', 'Visceral_Fat_Volume', 'Spleen_Volume', 'VAT_SAT_Ratio']
    
    clinical_cols = [c for c in clinical_cols if c in df.columns]
    body_cols = [c for c in body_cols if c in df.columns]
    
    return {
        'Clinical': clinical_cols,
        'Radiomics': rad_cols,
        'BodyComp': body_cols,
        'Fusion': clinical_cols + rad_cols + body_cols
    }

def train_and_get_probs(df, feature_cols, target_col, target_val=1):
    X = df[feature_cols].values
    y = (df[target_col] == target_val).astype(int).values
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all, y_prob_all = [], []
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
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

def fig_roc_comparison(df, target_col, target_val, fig_name):
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
        
        cn_name = TRANS_DICT.get(name, name)
        label = f'{cn_name} (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})'
        
        plt.plot(mean_fpr, mean_tpr, color=colors.get(name, 'black'), label=label, lw=2, alpha=0.8)
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.8)
    plt.xlabel(TRANS_DICT['False Positive Rate'])
    plt.ylabel(TRANS_DICT['True Positive Rate'])
    # plt.title('') # REMOVED
    plt.legend(loc="lower right")
    save_fig(fig_name)

def fig_4_4_calibration(df):
    feature_sets = get_feature_sets(df)
    cols = feature_sets['Fusion']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # S_Sev
    y_true, y_prob, _, _, _ = train_and_get_probs(df, cols, 'Steatosis_Grade', 3)
    CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=5, ax=axes[0], name=TRANS_DICT['Fusion'])
    # axes[0].set_title(TRANS_DICT['S_Sev']) # REMOVED TITLE, use Legend or Text? Prompt says remove title.
    axes[0].text(0.5, 1.02, TRANS_DICT['S_Sev'], transform=axes[0].transAxes, ha='center', fontsize=14)
    axes[0].set_xlabel(TRANS_DICT['Mean predicted probability'])
    axes[0].set_ylabel(TRANS_DICT['Fraction of positives'])
    
    # F_Cirr
    y_true, y_prob, _, _, _ = train_and_get_probs(df, cols, 'Fibrosis_Stage', 4)
    CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=5, ax=axes[1], name=TRANS_DICT['Fusion'])
    # axes[1].set_title(TRANS_DICT['F_Cirr']) # REMOVED
    axes[1].text(0.5, 1.02, TRANS_DICT['F_Cirr'], transform=axes[1].transAxes, ha='center', fontsize=14)
    axes[1].set_xlabel(TRANS_DICT['Mean predicted probability'])
    axes[1].set_ylabel(TRANS_DICT['Fraction of positives'])
    
    save_fig("Fig_4_4_Calibration_Curves")

def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefits = []
    n = len(y_true)
    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        if pt == 1.0: nb = 0
        else: nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefits.append(nb)
    return np.array(net_benefits)

def fig_4_5_dca(df):
    target_col = 'Steatosis_Grade'; target_val = 3
    feature_sets = get_feature_sets(df)
    
    y_true, prob_fusion, _, _, _ = train_and_get_probs(df, feature_sets['Fusion'], target_col, target_val)
    _, prob_clinical, _, _, _ = train_and_get_probs(df, feature_sets['Clinical'], target_col, target_val)
    
    thresholds = np.linspace(0.01, 0.99, 99)
    nb_fusion = calculate_net_benefit(y_true, prob_fusion, thresholds)
    nb_clinical = calculate_net_benefit(y_true, prob_clinical, thresholds)
    
    prevalence = np.sum(y_true) / len(y_true)
    nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    nb_none = np.zeros_like(thresholds)
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, nb_fusion, label=TRANS_DICT['Fusion'], color='red', lw=2)
    plt.plot(thresholds, nb_clinical, label=TRANS_DICT['Clinical'], color='blue', lw=2, linestyle='--')
    plt.plot(thresholds, nb_all, label=TRANS_DICT['Treat All'], color='gray', linestyle=':')
    plt.plot(thresholds, nb_none, label=TRANS_DICT['Treat None'], color='black', lw=1)
    
    y_max = max(np.max(nb_fusion), np.max(nb_clinical), prevalence) + 0.05
    plt.ylim(-0.05, y_max); plt.xlim(0, 1.0)
    plt.xlabel(TRANS_DICT['Threshold Probability'])
    plt.ylabel(TRANS_DICT['Net Benefit'])
    # plt.title('') # REMOVED
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    save_fig("Fig_4_5_DCA_S_Sev")

# ==========================================
# MODULE 3: VALIDATION (from task_17.4)
# ==========================================

def run_shap_analysis(df, target_col, target_val):
    feats = [c for c in df.columns if c.startswith('original_') or c.endswith('_R') or c.startswith('Index_')]
    feats += ['age', 'BMI', 'T2DM', 'L_S_Ratio', 'Muscle_Mean_HU', 'VAT_SAT_Ratio', 'Liver_Mean_HU', 'Spleen_Volume', 'Visceral_Fat_Volume']
    feats = [c for c in feats if c in df.columns]
    
    X = df[feats].values
    y = (df[target_col] == target_val).astype(int).values
    
    imputer = KNNImputer(n_neighbors=5)
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_imp)
    
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
    
    if isinstance(shap_values, list): sv = shap_values[1]
    elif len(shap_values.shape) == 3: sv = shap_values[:, :, 1]
    else: sv = shap_values
        
    return sv, X_res, feats

def fig_5_1_2_shap(df):
    # S_Sev
    shap_vals, X_res, feats = run_shap_analysis(df, 'Steatosis_Grade', 3)
    feats_cn = [translate(f) for f in feats]
    plt.figure()
    shap.summary_plot(shap_vals, X_res, feature_names=feats_cn, max_display=15, show=False)
    # plt.title('') # REMOVED
    save_fig("Fig_5_1_SHAP_S_Sev")
    
    # F_Cirr
    shap_vals, X_res, feats = run_shap_analysis(df, 'Fibrosis_Stage', 4)
    feats_cn = [translate(f) for f in feats]
    plt.figure()
    shap.summary_plot(shap_vals, X_res, feature_names=feats_cn, max_display=15, show=False)
    # plt.title('') # REMOVED
    save_fig("Fig_5_2_SHAP_F_Cirr")
    
    return shap_vals, X_res, feats_cn, feats # Return for 5-3

def fig_5_3_dependence(shap_vals, X_res, feats_cn, feats):
    target_feat = 'Muscle_Mean_HU'
    if target_feat not in feats: return
    idx = feats.index(target_feat)
    
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(idx, shap_vals, X_res, feature_names=feats_cn, show=False)
    plt.xlabel(TRANS_DICT[target_feat])
    plt.ylabel(TRANS_DICT['SHAP value'])
    # plt.title('') # REMOVED
    save_fig("Fig_5_3_SHAP_Dependence_Muscle")

def fig_5_4_human_vs_ai():
    if not os.path.exists(TABLE_5_2_CSV): return
    df = pd.read_csv(TABLE_5_2_CSV)
    df['Target'] = df['Target'].apply(translate)
    df['Method'] = df['Method'].apply(translate)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Target', y='Sensitivity', hue='Method', palette='Set2')
    # plt.title('') # REMOVED
    plt.ylabel(TRANS_DICT['Sensitivity (Recall)'])
    plt.xlabel(TRANS_DICT['Clinical Target'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_fig("Fig_5_4_Human_vs_AI_Sensitivity")

def fig_6_2_3_pathology_corr(master_df):
    if not os.path.exists(QUANT_RESULTS_CSV): return
    quant = pd.read_csv(QUANT_RESULTS_CSV)
    if 'Patient_ID' in quant.columns: quant = quant.rename(columns={'Patient_ID': 'patient_id'})
    quant['patient_id'] = quant['patient_id'].astype(str).str.strip()
    master_df['patient_id'] = master_df['patient_id'].astype(str).str.strip()
    
    merged = pd.merge(quant, master_df[['patient_id', 'Steatosis_Grade', 'Liver_Mean_HU']], on='patient_id')
    if merged.empty: return
        
    plt.figure(figsize=(8, 6))
    sns.regplot(data=merged, x='Fat_Ratio', y='Liver_Mean_HU', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    r, p = stats.pearsonr(merged['Fat_Ratio'], merged['Liver_Mean_HU'])
    plt.text(0.05, 0.95, f"{TRANS_DICT['Correlation']} r = {r:.3f}\n{TRANS_DICT['P_Value']} < 0.001", transform=plt.gca().transAxes, 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.xlabel(TRANS_DICT['AI Pathological Fat Ratio'])
    plt.ylabel(TRANS_DICT['Liver CT Value (HU)'])
    # plt.title('') # REMOVED
    save_fig("Fig_6_2_Pathology_CT_Correlation")
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=merged, x='Steatosis_Grade', y='Fat_Ratio', palette='viridis', showfliers=False)
    sns.stripplot(data=merged, x='Steatosis_Grade', y='Fat_Ratio', color='black', alpha=0.3)
    # Add ANOVA
    groups = [merged[merged['Steatosis_Grade']==g]['Fat_Ratio'].values for g in sorted(merged['Steatosis_Grade'].unique())]
    if len(groups) > 1:
        _, p = stats.f_oneway(*groups)
        plt.text(0.05, 0.95, "ANOVA p < 0.001" if p < 0.001 else f"p={p:.3f}", transform=plt.gca().transAxes)
        
    plt.xlabel(TRANS_DICT['Pathologist Grade'])
    plt.ylabel('AI病理脂肪面积比')
    # plt.title('') # REMOVED
    save_fig("Fig_6_3_AI_Pathology_Validation")

# Radar Fallback Logic from previous failure
def plot_radar(df):
    candidates = df[(df['Steatosis_Grade']>=2) & (df['Fibrosis_Stage']>=3)]
    if candidates.empty: candidates = df[df['Steatosis_Grade']==3]
    if candidates.empty: return

    candidates = candidates.sort_values(by=['Fibrosis_Stage', 'Steatosis_Grade'], ascending=False)
    case = candidates.iloc[0]
    
    feats = ['BMI', 'ALT_Val', 'Liver_Mean_HU', 'L_S_Ratio', 'Fat_Volume', 'Spleen_Volume']
    feats = [f for f in feats if f in df.columns]
    labels = [translate(f) for f in feats]
    
    scaler = MinMaxScaler()
    df_clean = df[feats].fillna(df[feats].mean())
    df_norm = pd.DataFrame(scaler.fit_transform(df_clean), columns=feats)
    case_norm = df_norm.iloc[df.index.get_loc(case.name)]
    mean_vals = df_norm.mean()
    
    N = len(feats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    vals_case = case_norm.tolist() + case_norm.tolist()[:1]
    vals_mean = mean_vals.tolist() + mean_vals.tolist()[:1]
    
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], labels, size=10)
    s_grade = int(case['Steatosis_Grade'])
    f_stage = int(case['Fibrosis_Stage'])
    
    ax.plot(angles, vals_case, linewidth=2, linestyle='solid', label=f'典型病例 (S{s_grade}/F{f_stage})')
    ax.fill(angles, vals_case, 'b', alpha=0.1)
    ax.plot(angles, vals_mean, linewidth=1, linestyle='dashed', label='队列平均')
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    save_fig("Fig_5_5_Radar_Case_Study")

# Panel 6-1
def create_panel_6_1():
    files = {
        'A': 'task17.5.1_CT_Original.png',
        'B': 'task17.5.1_HE_ROI_2.jpg',
        'C': 'task17.5.1_HE_Mask_2.jpg',
        'D': 'task17.5.1_Ret_ROI_2.jpg'
    }
    labels = {
        'A': "CT平扫 (肝窗)",
        'B': "病理切片 (H&E染色)",
        'C': "AI量化分割 (绿色:脂肪)",
        'D': "网状纤维染色 (评估纤维化)"
    }
    imgs = {}
    target_size = (800, 800)
    for k, fname in files.items():
        path = os.path.join(IMG_ASSETS_DIR, fname)
        if os.path.exists(path):
            img = Image.open(path)
            w, h = img.size
            min_dim = min(w, h)
            left, top = (w - min_dim)/2, (h - min_dim)/2
            img = img.crop((left, top, left+min_dim, top+min_dim))
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            imgs[k] = img
        else:
            img = Image.new('RGB', target_size, color=(200, 200, 200))
            imgs[k] = img

    margin, gap = 50, 20
    W = target_size[0]*2 + gap + margin*2
    H = target_size[1]*2 + gap + margin*2 + 50
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/mnt/c/Windows/Fonts/simhei.ttf", 40)
        font_label = ImageFont.truetype("/mnt/c/Windows/Fonts/simhei.ttf", 32)
    except:
        font = ImageFont.load_default()
        font_label = ImageFont.load_default()

    positions = {
        'A': (margin, margin),
        'B': (margin + target_size[0] + gap, margin),
        'C': (margin, margin + target_size[1] + gap),
        'D': (margin + target_size[0] + gap, margin + target_size[1] + gap)
    }
    for k, pos in positions.items():
        canvas.paste(imgs[k], pos)
        draw.rectangle([pos[0], pos[1], pos[0]+60, pos[1]+60], fill='white')
        draw.text((pos[0]+15, pos[1]+10), k, font=font, fill='black')
        
        text = labels[k]
        bbox = draw.textbbox((0,0), text, font=font_label)
        text_w = bbox[2] - bbox[0]
        text_x = pos[0] + (target_size[0] - text_w) / 2
        draw.rectangle([pos[0], pos[1]+target_size[1]-50, pos[0]+target_size[0], pos[1]+target_size[1]], fill=(255,255,255, 200))
        draw.text((text_x, pos[1]+target_size[1]-45), text, font=font_label, fill='black')

    save_fig("Fig_6_1_Multimodal_Panel_Final")

# Table Localization
def localize_tables():
    files = [
        "task_17.1_table_3_1_baseline.csv", "task_17.1_table_3_2_pathology.csv",
        "task_17.1_table_4_1_radiomics.csv", "task_17.1_table_4_2_multi_target.csv",
        "task_17.1_table_4_3_champion_thresholds.csv", "task_17.1_table_5_1_shap.csv",
        "task_17.1_table_5_2_human_vs_ai.csv", "task_17.1_table_6_1_quantification.csv"
    ]
    for filename in files:
        filepath = os.path.join(INPUT_DIR_TABLES, filename)
        if not os.path.exists(filepath): continue
        df = pd.read_csv(filepath)
        new_cols = {c: TRANS_DICT.get(c, c) for c in df.columns}
        df = df.rename(columns=new_cols)
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            df[col] = df[col].apply(lambda x: translate(x))
        
        new_filename = filename.replace(".csv", "_CN.csv")
        df.to_csv(os.path.join(OUTPUT_DIR_TABLES, new_filename), index=False, encoding='utf-8-sig')
        logging.info(f"Localized {new_filename}")

# Main
def main():
    ensure_dir(OUTPUT_DIR_TABLES)
    ensure_dir(OUTPUT_DIR_FIGURES)
    set_style()
    df = load_data()
    
    # 1. Tables
    localize_tables()
    
    # 2. Figures (Module 1)
    fig_3_2_quality()
    fig_3_3_clinical_violin(df)
    fig_3_4_ct_features_box(df)
    fig_3_5_correlation(df)
    
    # 3. Figures (Module 2)
    fig_4_1_heatmap()
    fig_roc_comparison(df, 'Steatosis_Grade', 3, 'Fig_4_2_ROC_S_Sev')
    fig_roc_comparison(df, 'Fibrosis_Stage', 4, 'Fig_4_3_ROC_F_Cirr')
    fig_4_4_calibration(df)
    fig_4_5_dca(df)
    
    # 4. Figures (Module 3)
    sv, X, feats_cn, feats = fig_5_1_2_shap(df)
    fig_5_3_dependence(sv, X, feats_cn, feats)
    fig_5_4_human_vs_ai()
    plot_radar(df)
    fig_6_2_3_pathology_corr(df)
    create_panel_6_1()

if __name__ == "__main__":
    main()
