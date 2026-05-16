import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE

# --- Setup ---
BASE_DIR = r"D:\mWork\paper0"
INPUT_CSV = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "20260323")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 中文支持配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 特征映射字典 (维持 v2 版本) ---
FEAT_MAP = {
    'age': '年龄',
    'BMI': 'BMI 指数',
    'T2DM': '2型糖尿病',
    'RDW-SD_R': '红细胞分布宽度-SD (异常)',
    'MO#_R': '单核细胞绝对数 (异常)',
    'BA#_R': '嗜碱性粒细胞绝对数 (异常)',
    'EO%_R': '嗜酸性粒细胞百分比 (异常)',
    'WBC_R': '白细胞计数 (异常)',
    'PLT_R': '血小板计数 (异常)',
    'RBC_R': '红细胞计数 (异常)',
    'AST/ALT_R': 'AST/ALT 比值 (异常)',
    'ALT_R': '谷丙转氨酶 (ALT) 异常',
    'AST_R': '谷草转氨酶 (AST) 异常',
    'PA_R': '前白蛋白 (异常)',
    'HCT_R': '红细胞压积 (异常)',
    'LDH_R': '乳酸脱氢酶 (异常)',
    'TBA_R': '总胆汁酸 (异常)',
    'MO%_R': '单核细胞百分比 (异常)',
    'Index_AST_ALT': 'AST/ALT 指数',
    'Index_SII': '系统免疫炎症指数 (SII)',
    'Index_APRI': 'APRI 评分 (纤维化)',
    'Index_FIB4': 'FIB-4 指数',
    'Index_PLR': '血小板/淋巴细胞比值 (PLR)',
    'L_S_Ratio': '肝脾密度比',
    'Muscle_Mean_HU': '骨骼肌密度 (HU)',
    'Liver_Mean_HU': '肝脏平均密度 (HU)',
    'Spleen_Volume': '脾脏体积',
    'Visceral_Fat_Volume': '内脏脂肪体积',
    'Fat_Volume': '皮下脂肪体积',
    'VAT_SAT_Ratio': '内脏/皮下脂肪比',
    'original_firstorder_10Percentile': '一阶特征: 10%分位数',
    'original_firstorder_90Percentile': '一阶特征: 90%分位数',
    'original_firstorder_Median': '一阶特征: 中位数',
    'original_firstorder_Mean': '一阶特征: 平均值',
    'original_firstorder_RootMeanSquared': '一阶特征: 均方根 (RMS)',
    'original_firstorder_TotalEnergy': '一阶特征: 总能量',
    'original_firstorder_Energy': '一阶特征: 能量',
    'original_firstorder_Skewness': '一阶特征: 偏度',
    'original_firstorder_Kurtosis': '一阶特征: 峰度',
    'original_glcm_ClusterShade': 'GLCM纹理: 聚类阴影',
    'original_ngtdm_Strength': '纹理特征: 强度 (Strength)',
    'original_glszm_SizeZoneNonUniformityNormalized': 'GLSZM纹理: 归一化区域大小不均匀性',
    'original_glrlm_GrayLevelVariance': 'GLRLM纹理: 灰度方差',
}

def translate(f):
    if f in FEAT_MAP: return FEAT_MAP[f]
    clean = f.replace('original_', '').replace('_', ' ')
    if 'firstorder' in clean: return f"一阶: {clean.split(' ')[-1]}"
    if 'glcm' in clean: return f"GLCM纹理: {clean.split(' ')[-1]}"
    if 'glszm' in clean: return f"GLSZM纹理: {clean.split(' ')[-1]}"
    return f

def run_v3_alignment():
    df = pd.read_csv(INPUT_CSV)
    df = df[df['L_S_Ratio'].notna() | df['original_firstorder_Energy'].notna()].copy()
    
    feats = [c for c in df.columns if c.startswith('original_') or c.endswith('_R') or c.startswith('Index_')]
    feats += ['age', 'BMI', 'T2DM', 'L_S_Ratio', 'Muscle_Mean_HU', 'Liver_Mean_HU', 'Spleen_Volume', 'Visceral_Fat_Volume', 'Fat_Volume', 'VAT_SAT_Ratio']
    feats = [c for c in feats if c in df.columns]
    
    targets = {
        '重度脂肪变 (S=3)': (df['Steatosis_Grade'] == 3).astype(int),
        '肝硬化 (F=4)': (df['Fibrosis_Stage'] == 4).astype(int)
    }
    
    all_importance = []
    
    for name, y in targets.items():
        print(f"正在生成 {name} 的最终学术图表...")
        X_raw = df[feats].values
        X_imputed = KNNImputer(n_neighbors=5).fit_transform(X_raw)
        X_scaled = StandardScaler().fit_transform(X_imputed)
        
        k = min(int(np.sum(y)) - 1, 3)
        X_res, y_res = SMOTE(random_state=42, k_neighbors=k).fit_resample(X_scaled, y)
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_res, y_res)
        
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_res, check_additivity=False)
        if isinstance(shap_values, list): shap_vals_pos = shap_values[1]
        elif len(shap_values.shape) == 3: shap_vals_pos = shap_values[:, :, 1]
        else: shap_vals_pos = shap_values
        
        feats_cn = [translate(f) for f in feats]
        X_res_df = pd.DataFrame(X_res, columns=feats_cn)
        
        # --- 绘图 (学术精修版) ---
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals_pos, X_res_df, show=False, max_display=15)
        # 汉化 X 轴
        plt.xlabel("SHAP值 (对模型输出的影响)", fontsize=14)
        # 移除标题 (不再使用 plt.title)
        
        file_tag = "S3" if "S=3" in name else "F4"
        plot_path = os.path.join(OUTPUT_DIR, f"SHAP_Summary_{file_tag}_Final_CN.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # 收集数据
        mean_abs_shap = np.mean(np.abs(shap_vals_pos), axis=0)
        for f_idx, val in enumerate(mean_abs_shap):
            all_importance.append({
                '预测靶点': name,
                '特征': feats_cn[f_idx],
                '平均SHAP值': round(val, 4)
            })
            
    # --- 保存 CSV 并仅保留 TOP 15 ---
    result_df = pd.DataFrame(all_importance)
    result_df = result_df.sort_values(by=['预测靶点', '平均SHAP值'], ascending=[False, False])
    
    # 每个靶点保留前 15
    top15_df = result_df.groupby('预测靶点').head(15)
    
    csv_path = os.path.join(OUTPUT_DIR, "SHAP_Importance_Top15_Aligned_CN.csv")
    top15_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"学术图表生成完成！\nCSV: {csv_path}\nPNGs: SHAP_Summary_S3/F4_Final_CN.png")

if __name__ == "__main__":
    run_v3_alignment()
