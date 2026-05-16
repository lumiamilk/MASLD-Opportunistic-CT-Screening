"""
绘制S≥2和S=3的对比ROC图
图9：脂肪变性预测模型ROC曲线对比
"""
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import logging

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
MASTER_CSV = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "thesis_assets", "figures")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 中文翻译字典 ---
TRANS_DICT = {
    "True Positive Rate": "敏感度 (Sensitivity)",
    "False Positive Rate": "1 - 特异度 (1 - Specificity)",
    "Clinical": "临床模型",
    "Radiomics": "影像组学模型",
    "BodyComp": "体成分模型",
    "Fusion": "融合模型",
}

# --- Utils ---
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_style():
    """设置中文字体和学术风格"""
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
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    
    if os.path.exists(font_path):
        plt.rcParams['font.sans-serif'] = [prop.get_name()] + plt.rcParams['font.sans-serif']

def load_data():
    """加载主数据集"""
    try:
        df = pd.read_csv(MASTER_CSV, encoding='utf-8-sig')
    except:
        df = pd.read_csv(MASTER_CSV, encoding='gb18030')
    return df

def get_fusion_cohort(df):
    """获取融合队列（有CT和影像组学数据的样本）"""
    mask = df['L_S_Ratio'].notna() | df['original_firstorder_Energy'].notna()
    return df[mask].copy()

def get_feature_sets(df):
    """定义特征集"""
    rad_cols = [c for c in df.columns if c.startswith('original_')]
    clinical_cols = ['age', 'sex', 'BMI', 'T2DM', 'High_Blood_pressure', 
                     'ALT_Val', 'AST_Val', 'PLT_Val', 'TG_Val']
    body_cols = ['Liver_Mean_HU', 'L_S_Ratio', 'Muscle_Mean_HU', 
                 'Fat_Volume', 'Visceral_Fat_Volume', 'Spleen_Volume', 'VAT_SAT_Ratio']
    
    clinical_cols = [c for c in clinical_cols if c in df.columns]
    body_cols = [c for c in body_cols if c in df.columns]
    
    return {
        'Clinical': clinical_cols,
        'Radiomics': rad_cols,
        'BodyComp': body_cols,
        'Fusion': clinical_cols + rad_cols + body_cols
    }

def train_and_get_probs(df, feature_cols, target_col, target_val=1, comparison='eq'):
    """
    训练模型并获取预测概率
    
    Parameters:
    -----------
    comparison : str
        'eq' 表示等于 (target_col == target_val)
        'ge' 表示大于等于 (target_col >= target_val)
    """
    X = df[feature_cols].values
    
    if comparison == 'ge':
        y = (df[target_col] >= target_val).astype(int).values
    else:
        y = (df[target_col] == target_val).astype(int).values
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all = []
    y_prob_all = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # SMOTE k_neighbors check
        k = min(np.sum(y_train) - 1, 5)
        if k < 1:
            k = 1
        
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

def plot_single_roc(ax, df, target_col, target_val, comparison, title_label):
    """在指定axes上绘制单条ROC曲线组"""
    feature_sets = get_feature_sets(df)
    colors = {
        'Clinical': '#1f77b4',      # 蓝色
        'Radiomics': '#2ca02c',     # 绿色
        'BodyComp': '#ff7f0e',      # 橙色
        'Fusion': '#d62728'         # 红色
    }
    
    for name, cols in feature_sets.items():
        if not cols:
            continue
        
        _, _, tprs, aucs, mean_fpr = train_and_get_probs(
            df, cols, target_col, target_val, comparison
        )
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        cn_name = TRANS_DICT.get(name, name)
        label = f'{cn_name} (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})'
        
        ax.plot(mean_fpr, mean_tpr, color=colors.get(name, 'black'),
                label=label, lw=2, alpha=0.85)
    
    # 对角线
    ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='gray', alpha=0.7)
    
    # 设置标签
    ax.set_xlabel(TRANS_DICT['False Positive Rate'], fontsize=12)
    ax.set_ylabel(TRANS_DICT['True Positive Rate'], fontsize=12)
    
    # 添加子图标签（A/B）在左上角
    ax.text(0.02, 0.98, title_label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black'))
    
    # 图例
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    
    # 设置坐标轴范围
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')

def main():
    """主函数：生成图9"""
    ensure_dir(OUTPUT_DIR)
    set_style()
    
    df = load_data()
    fusion_df = get_fusion_cohort(df)
    
    logging.info(f"融合队列样本量: {len(fusion_df)}")
    logging.info(f"S≥2阳性样本: {(fusion_df['Steatosis_Grade'] >= 2).sum()}")
    logging.info(f"S=3阳性样本: {(fusion_df['Steatosis_Grade'] == 3).sum()}")
    
    # 创建图：两个子图并排
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    
    # 左图：S≥2（中重度脂肪变性）
    plot_single_roc(axes[0], fusion_df, 'Steatosis_Grade', 2, 'ge', 'A')
    
    # 右图：S=3（重度脂肪变性）
    plot_single_roc(axes[1], fusion_df, 'Steatosis_Grade', 3, 'eq', 'B')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(OUTPUT_DIR, "图9.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    logging.info(f"图片已保存: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    main()
