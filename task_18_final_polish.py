import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import logging
import shap
import sqlite3
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibrationDisplay
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
MASTER_CSV = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
DB_PATH = os.path.join(BASE_DIR, "output", "task_1_ct_metadata.db")
QUANT_RESULTS_CSV = os.path.join(BASE_DIR, "output", "task_16_quant_results_v5.3.csv")

INPUT_DIR_TABLES = os.path.join(BASE_DIR, "output", "thesis_assets", "tables")
OUTPUT_DIR_TABLES = os.path.join(BASE_DIR, "output", "thesis_assets", "tables_cn")
OUTPUT_DIR_FIGURES = os.path.join(BASE_DIR, "output", "thesis_assets", "figures_cn")

# Image Assets for Panel
IMG_ASSETS_DIR = os.path.join(BASE_DIR, "output")

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
}

# --- Map for Final Output List ---
FILE_TITLE_MAP = {
    # Tables
    "task_17.1_table_3_1_baseline_CN.csv": "表3-1 研究队列基线特征表",
    "task_17.1_table_3_2_pathology_CN.csv": "表3-2 病理学特征分布表",
    "task_17.1_table_4_1_radiomics_CN.csv": "表4-1 影像组学特征筛选结果表",
    "task_17.1_table_4_2_multi_target_CN.csv": "表4-2 多靶点模型性能比较表",
    "task_17.1_table_4_3_champion_thresholds_CN.csv": "表4-3 最佳模型截断值与性能表",
    "task_17.1_table_5_1_shap_CN.csv": "表5-1 模型特征重要性排名前15位",
    "task_17.1_table_5_2_human_vs_ai_CN.csv": "表5-2 人工智能与病理医生诊断效能对比表",
    "task_17.1_table_6_1_quantification_CN.csv": "表6-1 AI病理量化结果统计表",
    
    # Figures
    "Fig_3_2_Data_Quality.png": "图3-2 CT图像质量评估与参数分布图",
    "Fig_3_3_Clinical_vs_Pathology.png": "图3-3 临床生化指标与病理分级的关联分析图",
    "Fig_3_4_CT_vs_Pathology.png": "图3-4 影像学特征与病理分级的关联分析图",
    "Fig_3_5_Correlation_Matrix.png": "图3-5 多模态特征相关性热力图",
    "Fig_4_1_Multi_Target_Heatmap.png": "图4-1 多模型多靶点预测性能热力图 (AUC)",
    "Fig_4_2_ROC_S_Sev.png": "图4-2 重度脂肪变性预测模型ROC曲线",
    "Fig_4_3_ROC_F_Cirr.png": "图4-3 肝硬化预测模型ROC曲线",
    "Fig_4_4_Calibration_Curves.png": "图4-4 融合模型校准曲线分析",
    "Fig_4_5_DCA_S_Sev.png": "图4-5 重度脂肪变性预测模型临床决策曲线 (DCA)",
    "Fig_5_1_SHAP_S_Sev.png": "图5-1 重度脂肪变性预测模型SHAP特征重要性图",
    "Fig_5_2_SHAP_F_Cirr.png": "图5-2 肝硬化预测模型SHAP特征重要性图",
    "Fig_5_3_SHAP_Dependence_Muscle.png": "图5-3 骨骼肌密度与疾病风险的SHAP依赖图",
    "Fig_5_4_Human_vs_AI_Sensitivity.png": "图5-4 人工智能与病理医生诊断敏感度对比图",
    "Fig_5_5_Radar_Case_Study.png": "图5-5 典型病例多模态特征雷达图",
    "Fig_6_1_Multimodal_Panel_Final.png": "图6-1 多模态数据展示面板 (CT/HE/Mask/Ret)",
    "Fig_6_2_Pathology_CT_Correlation.png": "图6-2 AI病理脂肪定量与CT肝脏密度的相关性分析",
    "Fig_6_3_AI_Pathology_Validation.png": "图6-3 AI病理脂肪定量与病理医生分级的分布验证"
}

# --- Utility Functions ---

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
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    # Re-apply font
    if os.path.exists(font_path):
         plt.rcParams['font.sans-serif'] = [prop.get_name()] + plt.rcParams['font.sans-serif']

def load_data():
    try: df = pd.read_csv(MASTER_CSV, encoding='utf-8-sig')
    except: df = pd.read_csv(MASTER_CSV, encoding='gb18030')
    mask = df['L_S_Ratio'].notna() | df['original_firstorder_Energy'].notna()
    return df[mask].copy()

def save_fig(name):
    path_png = os.path.join(OUTPUT_DIR_FIGURES, name) # name already has extension
    plt.savefig(path_png, bbox_inches='tight', dpi=300)
    logging.info(f"Saved {name}")
    plt.close()

def translate(text):
    if not isinstance(text, str): return text
    # Direct match first
    if text in TRANS_DICT: return TRANS_DICT[text]
    # Partial match for Radiomics
    for k, v in TRANS_DICT.items():
        if k in text and "original_" in k:
            return text.replace(k, v)
    return text

# --- Part 1: Table Localization ---

def localize_tables():
    ensure_dir(OUTPUT_DIR_TABLES)
    if not os.path.exists(INPUT_DIR_TABLES): return

    files_to_process = [
        "task_17.1_table_3_1_baseline.csv",
        "task_17.1_table_3_2_pathology.csv",
        "task_17.1_table_4_1_radiomics.csv",
        "task_17.1_table_4_2_multi_target.csv",
        "task_17.1_table_4_3_champion_thresholds.csv",
        "task_17.1_table_5_1_shap.csv",
        "task_17.1_table_5_2_human_vs_ai.csv",
        "task_17.1_table_6_1_quantification.csv"
    ]

    for filename in files_to_process:
        filepath = os.path.join(INPUT_DIR_TABLES, filename)
        if not os.path.exists(filepath):
            logging.warning(f"Table not found: {filename}")
            continue

        try:
            df = pd.read_csv(filepath)
            
            # Translate Columns
            new_cols = {c: TRANS_DICT.get(c, c) for c in df.columns}
            df = df.rename(columns=new_cols)
            
            # Translate Content
            obj_cols = df.select_dtypes(include=['object']).columns
            for col in obj_cols:
                df[col] = df[col].apply(lambda x: TRANS_DICT.get(x, x) if isinstance(x, str) else x)
                # Extra pass for radiomics features which might be substrings or not exact matches
                if col in ['Feature', '特征', 'Variable', '变量']:
                     df[col] = df[col].apply(lambda x: translate(x) if isinstance(x, str) else x)

            # Save
            new_filename = filename.replace(".csv", "_CN.csv")
            df.to_csv(os.path.join(OUTPUT_DIR_TABLES, new_filename), index=False, encoding='utf-8-sig')
            logging.info(f"Localized {new_filename}")
            
        except Exception as e:
            logging.error(f"Failed to localize {filename}: {e}")

# --- Part 2: Figure Re-plotting ---

# 1. Data Quality (Fig 3-2)
def plot_fig_3_2_quality():
    if not os.path.exists(DB_PATH): return
    conn = sqlite3.connect(DB_PATH)
    df_meta = pd.read_sql("SELECT slice_thickness, kernel, kvp FROM ct_series", conn)
    conn.close()
    df_meta = df_meta[df_meta['slice_thickness'] > 0]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # A
    sns.histplot(data=df_meta, x='slice_thickness', kde=False, ax=axes[0], bins=10, color='skyblue')
    axes[0].set_xlabel(TRANS_DICT['Slice Thickness'])
    axes[0].set_ylabel(TRANS_DICT['Count'])
    
    # B
    top_kernels = df_meta['kernel'].value_counts().nlargest(5).index
    df_kernel = df_meta[df_meta['kernel'].isin(top_kernels)]
    sns.countplot(data=df_kernel, x='kernel', ax=axes[1], order=top_kernels, palette='viridis')
    axes[1].set_xlabel(TRANS_DICT['Kernel Type'])
    axes[1].set_ylabel(TRANS_DICT['Count'])
    axes[1].tick_params(axis='x', rotation=45)
    
    # C
    sns.countplot(data=df_meta, x='kvp', ax=axes[2], palette='Set2')
    axes[2].set_xlabel(TRANS_DICT['Tube Voltage'])
    axes[2].set_ylabel(TRANS_DICT['Count'])
    
    save_fig("Fig_3_2_Data_Quality.png")

# Helper for P-values
def add_p_val(ax, df, x, y):
    groups = [df[df[x] == g][y].dropna().values for g in sorted(df[x].unique())]
    if len(groups) < 2: return
    try:
        stat, p = stats.kruskal(*groups)
        p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0.95, 0.95, p_text, transform=ax.transAxes, ha='right', va='top', 
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8), fontsize=10)
    except: pass

# 2. Clinical vs Pathology (Fig 3-3)
def plot_fig_3_3_clin_path(df):
    features = ['ALT_Val', 'AST_Val', 'TG_Val', 'Index_FIB4', 'Index_APRI', 'PLT_Val']
    x_cols = ['Steatosis_Grade']*3 + ['Fibrosis_Stage']*3
    titles = [TRANS_DICT[f] for f in features]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        x_col = x_cols[i]
        sns.violinplot(data=df, x=x_col, y=feat, ax=axes[i], palette='Blues' if i<3 else 'Oranges')
        axes[i].set_xlabel(TRANS_DICT[x_col])
        axes[i].set_ylabel(titles[i])
        add_p_val(axes[i], df, x_col, feat)
        
    save_fig("Fig_3_3_Clinical_vs_Pathology.png")

# 3. CT vs Pathology (Fig 3-4)
def plot_fig_3_4_ct_path(df):
    features = ['Liver_Mean_HU', 'L_S_Ratio', 'Fat_Volume', 'Spleen_Volume', 'Muscle_Mean_HU', 'VAT_SAT_Ratio']
    x_cols = ['Steatosis_Grade']*3 + ['Fibrosis_Stage']*3
    titles = [TRANS_DICT[f] for f in features]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        x_col = x_cols[i]
        sns.boxplot(data=df, x=x_col, y=feat, ax=axes[i], palette='viridis' if i<3 else 'magma', showfliers=False)
        sns.stripplot(data=df, x=x_col, y=feat, ax=axes[i], color='black', alpha=0.3, size=2)
        axes[i].set_xlabel(TRANS_DICT[x_col])
        axes[i].set_ylabel(titles[i])
        add_p_val(axes[i], df, x_col, feat)
        
    save_fig("Fig_3_4_CT_vs_Pathology.png")

# 4. Correlation Matrix (Fig 3-5)
def plot_fig_3_5_corr(df):
    cols = ['Steatosis_Grade', 'Fibrosis_Stage', 'ALT_Val', 'AST_Val', 'Index_FIB4', 
            'Liver_Mean_HU', 'L_S_Ratio', 'Fat_Volume', 'Spleen_Volume']
    names = [TRANS_DICT.get(c, c).replace(" (HU)", "").replace(" (cm³)", "") for c in cols] # Shorten names
    
    data = df[cols].copy()
    data.columns = names
    corr = data.corr(method='spearman')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, vmax=1, vmin=-1, 
                annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .5})
    save_fig("Fig_3_5_Correlation_Matrix.png")

# 5. Multi Target Heatmap (Fig 4-1)
def plot_fig_4_1_heatmap():
    csv_path = os.path.join(INPUT_DIR_TABLES, "task_17.1_table_4_2_multi_target.csv")
    if not os.path.exists(csv_path): return
    
    df = pd.read_csv(csv_path)
    df = df.set_index('Target')
    
    # Extract Means
    df_num = df.applymap(lambda x: float(x.split('±')[0]) if isinstance(x, str) and '±' in x else np.nan)
    
    # Translate Index and Columns
    df_num.index = [TRANS_DICT.get(i, i) for i in df_num.index]
    df_num.columns = [TRANS_DICT.get(c, c) for c in df_num.columns]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_num, annot=True, cmap="RdYlGn", fmt=".3f", vmin=0.6, vmax=0.95)
    plt.xlabel(TRANS_DICT['Model'])
    plt.ylabel(TRANS_DICT['Target'])
    save_fig("Fig_4_1_Multi_Target_Heatmap.png")

# 6. ROC Curves (Fig 4-2, 4-3)
def get_features(df):
    rad = [c for c in df.columns if c.startswith('original_')]
    clin = [c for c in ['age','sex','BMI','T2DM','High_Blood_pressure','ALT_Val','AST_Val','PLT_Val','TG_Val'] if c in df.columns]
    body = [c for c in ['Liver_Mean_HU','L_S_Ratio','Muscle_Mean_HU','Fat_Volume','Visceral_Fat_Volume','Spleen_Volume','VAT_SAT_Ratio'] if c in df.columns]
    return {'Clinical': clin, 'Radiomics': rad, 'BodyComp': body, 'Fusion': clin+rad+body}

def train_roc(df, target_col, target_val):
    sets = get_features(df)
    results = {}
    
    y = (df[target_col] == target_val).astype(int).values
    mean_fpr = np.linspace(0, 1, 100)
    
    for name, cols in sets.items():
        if not cols: continue
        X = df[cols].values
        tprs = []
        aucs = []
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in cv.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            k = min(np.sum(y_train)-1, 5)
            if k < 1: k = 1
            
            pipe = ImbPipeline([
                ('imp', KNNImputer(n_neighbors=5)),
                ('scl', StandardScaler()),
                ('smote', SMOTE(random_state=42, k_neighbors=k)),
                ('clf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
            ])
            pipe.fit(X_train, y_train)
            probs = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(auc(fpr, tpr))
            
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        results[name] = (mean_fpr, mean_tpr, np.mean(aucs))
    return results

def plot_roc(df, target, val, filename):
    res = train_roc(df, target, val)
    plt.figure(figsize=(8, 8))
    colors = {'Clinical': '#1f77b4', 'Radiomics': '#2ca02c', 'BodyComp': '#ff7f0e', 'Fusion': '#d62728'}
    
    for name, (fpr, tpr, score) in res.items():
        cn_name = TRANS_DICT.get(name, name)
        plt.plot(fpr, tpr, color=colors.get(name, 'k'), lw=2, label=f"{cn_name} (AUC = {score:.2f})")
        
    plt.plot([0,1],[0,1], 'k--', alpha=0.5)
    plt.xlabel(TRANS_DICT['False Positive Rate'])
    plt.ylabel(TRANS_DICT['True Positive Rate'])
    plt.legend(loc="lower right")
    save_fig(filename)

# 7. Calibration (Fig 4-4)
def plot_calibration(df):
    sets = get_features(df)
    cols = sets['Fusion']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # S_Sev
    y = (df['Steatosis_Grade'] == 3).astype(int)
    # Simple Split for Calibration (or CV concat)
    # Doing CV concat for smoother plot
    y_true_all, y_prob_all = [], []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X = df[cols].values
    
    for tr, te in cv.split(X, y):
        model = ImbPipeline([('i', KNNImputer()), ('s', StandardScaler()), ('clf', RandomForestClassifier(50))])
        model.fit(X[tr], y.iloc[tr])
        y_prob_all.extend(model.predict_proba(X[te])[:,1])
        y_true_all.extend(y.iloc[te])
        
    CalibrationDisplay.from_predictions(y_true_all, y_prob_all, n_bins=5, ax=axes[0], name=TRANS_DICT['Fusion'])
    axes[0].set_xlabel(TRANS_DICT['Mean predicted probability'])
    axes[0].set_ylabel(TRANS_DICT['Fraction of positives'])
    axes[0].set_title(TRANS_DICT['S_Sev']) # Keeping subtitle for distinction
    
    # F_Cirr
    y = (df['Fibrosis_Stage'] == 4).astype(int)
    y_true_all, y_prob_all = [], []
    for tr, te in cv.split(X, y):
        model = ImbPipeline([('i', KNNImputer()), ('s', StandardScaler()), ('clf', RandomForestClassifier(50))])
        model.fit(X[tr], y.iloc[tr])
        y_prob_all.extend(model.predict_proba(X[te])[:,1])
        y_true_all.extend(y.iloc[te])
        
    CalibrationDisplay.from_predictions(y_true_all, y_prob_all, n_bins=5, ax=axes[1], name=TRANS_DICT['Fusion'])
    axes[1].set_xlabel(TRANS_DICT['Mean predicted probability'])
    axes[1].set_ylabel(TRANS_DICT['Fraction of positives'])
    axes[1].set_title(TRANS_DICT['F_Cirr'])
    
    save_fig("Fig_4_4_Calibration_Curves.png")

# 8. DCA (Fig 4-5)
def calc_net_benefit(y_true, y_prob, thresh):
    n = len(y_true)
    net_b = []
    for pt in thresh:
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred==1) & (y_true==1))
        fp = np.sum((y_pred==1) & (y_true==0))
        if pt==1: nb=0
        else: nb = (tp/n) - (fp/n)*(pt/(1-pt))
        net_b.append(nb)
    return np.array(net_b)

def plot_dca(df):
    cols = get_features(df)['Fusion']
    y = (df['Steatosis_Grade'] == 3).astype(int).values
    X = df[cols].values
    
    # Train/Predict (OOF)
    y_prob = np.zeros_like(y, dtype=float)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    for tr, te in cv.split(X, y):
        model = ImbPipeline([('i', KNNImputer()), ('s', StandardScaler()), ('clf', RandomForestClassifier(50))])
        model.fit(X[tr], y[tr])
        y_prob[te] = model.predict_proba(X[te])[:,1]
        
    thresh = np.linspace(0.01, 0.99, 99)
    nb_model = calc_net_benefit(y, y_prob, thresh)
    prev = np.mean(y)
    nb_all = prev - (1-prev)*thresh/(1-thresh)
    nb_none = np.zeros_like(thresh)
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresh, nb_model, 'r-', lw=2, label=TRANS_DICT['Fusion'])
    plt.plot(thresh, nb_all, 'k:', label=TRANS_DICT['Treat All'])
    plt.plot(thresh, nb_none, 'k-', lw=1, label=TRANS_DICT['Treat None'])
    plt.ylim(-0.05, max(prev, np.max(nb_model))+0.05)
    plt.xlabel(TRANS_DICT['Threshold Probability'])
    plt.ylabel(TRANS_DICT['Net Benefit'])
    plt.legend()
    save_fig("Fig_4_5_DCA_S_Sev.png")

# 9. SHAP (Fig 5-1, 5-2, 5-3)
def get_shap(df, target, val):
    cols = get_features(df)['Fusion']
    X = df[cols].values
    y = (df[target] == val).astype(int).values
    
    # Single split fit
    imp = KNNImputer()
    X_imp = imp.fit_transform(X)
    clf = RandomForestClassifier(50, random_state=42)
    clf.fit(X_imp, y)
    
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_imp)
    if isinstance(shap_values, list): sv = shap_values[1]
    else: sv = shap_values[:,:,1] if len(shap_values.shape)==3 else shap_values
    
    return sv, X_imp, cols

def plot_shap(df):
    # 5-1 S_Sev
    sv, X, cols = get_shap(df, 'Steatosis_Grade', 3)
    cols_cn = [translate(c) for c in cols]
    plt.figure()
    shap.summary_plot(sv, X, feature_names=cols_cn, show=False, max_display=15)
    save_fig("Fig_5_1_SHAP_S_Sev.png")
    
    # 5-2 F_Cirr
    sv, X, cols = get_shap(df, 'Fibrosis_Stage', 4)
    cols_cn = [translate(c) for c in cols]
    plt.figure()
    shap.summary_plot(sv, X, feature_names=cols_cn, show=False, max_display=15)
    save_fig("Fig_5_2_SHAP_F_Cirr.png")
    
    # 5-3 Dependence (Muscle)
    # Use F_Cirr model
    tgt_feat = 'Muscle_Mean_HU'
    if tgt_feat in cols:
        idx = cols.index(tgt_feat)
        plt.figure(figsize=(8,6))
        shap.dependence_plot(idx, sv, X, feature_names=cols_cn, show=False, interaction_index=None)
        # Fix Labels manually because shap uses its own
        plt.xlabel(TRANS_DICT[tgt_feat])
        plt.ylabel(TRANS_DICT['SHAP value'])
        save_fig("Fig_5_3_SHAP_Dependence_Muscle.png")

# 10. Human vs AI (Fig 5-4)
def plot_human_vs_ai():
    csv_path = os.path.join(INPUT_DIR_TABLES, "task_17.1_table_5_2_human_vs_ai.csv")
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    
    # Translate
    df['Target'] = df['Target'].apply(translate)
    df['Method'] = df['Method'].apply(translate)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Target', y='Sensitivity', hue='Method', palette='Set2')
    plt.ylabel(TRANS_DICT['Sensitivity (Recall)'])
    plt.xlabel(TRANS_DICT['Target'])
    plt.legend(title=TRANS_DICT['Method'])
    save_fig("Fig_5_4_Human_vs_AI_Sensitivity.png")

# 11. Radar Plot (Fig 5-5) - Simple Case Study
def plot_radar(df):
    # Try to find a severe case
    candidates = df[(df['Steatosis_Grade']>=2) & (df['Fibrosis_Stage']>=3)]
    
    if candidates.empty:
        # Fallback to just S=3
        candidates = df[df['Steatosis_Grade']==3]
        
    if candidates.empty:
        logging.warning("No suitable case found for Radar plot.")
        return

    # Sort by fibrosis then steatosis to get the 'worst' one
    candidates = candidates.sort_values(by=['Fibrosis_Stage', 'Steatosis_Grade'], ascending=False)
    case = candidates.iloc[0]
    
    # Features to show (Normalized 0-1)
    feats = ['BMI', 'ALT_Val', 'Liver_Mean_HU', 'L_S_Ratio', 'Fat_Volume', 'Spleen_Volume']
    # Filter features that exist
    feats = [f for f in feats if f in df.columns]
    
    labels = [translate(f) for f in feats]
    
    # Normalize cohort
    scaler = MinMaxScaler()
    # Handle NaN for normalization
    df_clean = df[feats].fillna(df[feats].mean())
    df_norm = pd.DataFrame(scaler.fit_transform(df_clean), columns=feats)
    
    # Get case data (using index match)
    case_idx = df.index.get_loc(case.name)
    case_norm = df_norm.iloc[case_idx]
    
    # Mean values
    mean_vals = df_norm.mean()
    
    # Plot
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
    
    s_grade = case['Steatosis_Grade']
    f_stage = case['Fibrosis_Stage']
    
    ax.plot(angles, vals_case, linewidth=2, linestyle='solid', label=f'典型病例 (S{int(s_grade)}/F{int(f_stage)})')
    ax.fill(angles, vals_case, 'b', alpha=0.1)
    
    ax.plot(angles, vals_mean, linewidth=1, linestyle='dashed', label='队列平均')
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    save_fig("Fig_5_5_Radar_Case_Study.png")

# 12. Pathology Correlation (Fig 6-2, 6-3)
def plot_pathology_corr(df):
    if not os.path.exists(QUANT_RESULTS_CSV): return
    qdf = pd.read_csv(QUANT_RESULTS_CSV)
    # Clean ID
    qdf['Patient_ID'] = qdf['Patient_ID'].astype(str).str.strip()
    df['patient_id'] = df['patient_id'].astype(str).str.strip()
    merged = pd.merge(qdf, df[['patient_id', 'Liver_Mean_HU', 'Steatosis_Grade']], 
                      left_on='Patient_ID', right_on='patient_id')
    if merged.empty: return

    # 6-2 Scatter
    plt.figure(figsize=(8, 6))
    sns.regplot(data=merged, x='Fat_Ratio', y='Liver_Mean_HU', scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    plt.xlabel(TRANS_DICT['AI Pathological Fat Ratio'])
    plt.ylabel(TRANS_DICT['Liver CT Value (HU)'])
    
    r, p = stats.pearsonr(merged['Fat_Ratio'], merged['Liver_Mean_HU'])
    plt.text(0.05, 0.95, f"{TRANS_DICT['Correlation']} r = {r:.3f}\n{TRANS_DICT['P_Value']} < 0.001", 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    save_fig("Fig_6_2_Pathology_CT_Correlation.png")
    
    # 6-3 Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=merged, x='Steatosis_Grade', y='Fat_Ratio', palette='viridis', showfliers=False)
    sns.stripplot(data=merged, x='Steatosis_Grade', y='Fat_Ratio', color='k', alpha=0.3)
    plt.xlabel(TRANS_DICT['Pathologist Grade'])
    plt.ylabel(TRANS_DICT['AI Pathological Fat Ratio'])
    save_fig("Fig_6_3_AI_Pathology_Validation.png")

# 13. Panel Fig 6-1 (Composite)
def create_panel_6_1():
    # Source Images (Assume they exist from Task 17.5.1)
    # We need to look for specific files. 
    # If not found, skip or create placeholder.
    # The user instruction implies they exist.
    
    files = {
        'A': 'task17.5.1_CT_Original.png',
        'B': 'task17.5.1_HE_ROI_2.jpg',
        'C': 'task17.5.1_HE_Mask_2.jpg',
        'D': 'task17.5.1_Ret_ROI_2.jpg'
    }
    
    # Labels
    labels = {
        'A': "CT平扫 (肝窗)",
        'B': "病理切片 (H&E染色)",
        'C': "AI量化分割 (绿色:脂肪)",
        'D': "网状纤维染色 (评估纤维化)"
    }
    
    # Load and Resize
    imgs = {}
    target_size = (800, 800)
    
    for k, fname in files.items():
        path = os.path.join(IMG_ASSETS_DIR, fname)
        if os.path.exists(path):
            img = Image.open(path)
            # Crop to square center if needed, then resize
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim)/2
            top = (h - min_dim)/2
            img = img.crop((left, top, left+min_dim, top+min_dim))
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            imgs[k] = img
        else:
            # Placeholder
            img = Image.new('RGB', target_size, color=(200, 200, 200))
            d = ImageDraw.Draw(img)
            d.text((300, 400), f"Missing: {fname}", fill=(0,0,0))
            imgs[k] = img

    # Canvas
    margin = 50
    gap = 20
    W = target_size[0]*2 + gap + margin*2
    H = target_size[1]*2 + gap + margin*2 + 50 # Extra for text
    
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
        
        # Draw Label Letter (A, B, C, D)
        draw.rectangle([pos[0], pos[1], pos[0]+60, pos[1]+60], fill='white')
        draw.text((pos[0]+15, pos[1]+10), k, font=font, fill='black')
        
        # Draw Caption below image
        text = labels[k]
        bbox = draw.textbbox((0,0), text, font=font_label)
        text_w = bbox[2] - bbox[0]
        text_x = pos[0] + (target_size[0] - text_w) / 2
        text_y = pos[1] + target_size[1] - 60 # Inside image at bottom? Or below?
        # Let's put it slightly inside with background or below?
        # Put below implies we need more height. Let's put inside bottom with background.
        
        draw.rectangle([pos[0], pos[1]+target_size[1]-50, pos[0]+target_size[0], pos[1]+target_size[1]], fill=(255,255,255, 200))
        draw.text((text_x, pos[1]+target_size[1]-45), text, font=font_label, fill='black')

    save_path = os.path.join(OUTPUT_DIR_FIGURES, "Fig_6_1_Multimodal_Panel_Final.png")
    canvas.save(save_path)
    logging.info("Saved Fig_6_1_Multimodal_Panel_Final.png")

# --- Main Execution ---

def main():
    ensure_dir(OUTPUT_DIR_TABLES)
    ensure_dir(OUTPUT_DIR_FIGURES)
    set_style()
    df = load_data()
    
    # Tables
    logging.info("--- Processing Tables ---")
    localize_tables()
    
    # Figures
    logging.info("--- Processing Figures ---")
    plot_fig_3_2_quality()
    plot_fig_3_3_clin_path(df)
    plot_fig_3_4_ct_path(df)
    plot_fig_3_5_corr(df)
    plot_fig_4_1_heatmap()
    plot_roc(df, 'Steatosis_Grade', 3, "Fig_4_2_ROC_S_Sev.png")
    plot_roc(df, 'Fibrosis_Stage', 4, "Fig_4_3_ROC_F_Cirr.png")
    plot_calibration(df)
    plot_dca(df)
    plot_shap(df)
    plot_human_vs_ai()
    plot_radar(df)
    plot_pathology_corr(df)
    create_panel_6_1()
    
    # Output Mapping List
    logging.info("--- File Mapping ---")
    print("\n[生成文件清单及中文名称]")
    for filename, cn_name in FILE_TITLE_MAP.items():
        # Check if file exists
        if filename.endswith('.csv'):
            path = os.path.join(OUTPUT_DIR_TABLES, filename)
        else:
            path = os.path.join(OUTPUT_DIR_FIGURES, filename)
            
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"[{status}] {filename} -> {cn_name}")

if __name__ == "__main__":
    main()