"""
生成表7：人类医生与AI模型对比表（含S≥2）
输出纯中文CSV表格
"""
import pandas as pd
import numpy as np
import os
import json
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
MASTER_CSV = os.path.join(BASE_DIR, "output", "master_dataset_final.csv")
ORIGINAL_JSON = os.path.join(BASE_DIR, "data", "original_data.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "thesis_assets", "tables")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 中文翻译 ---
TRANS_DICT = {
    'S_Sev': '重度脂肪变性 (S=3)',
    'S_Ge2': '中重度脂肪变性 (S≥2)',
    'F_Cirr': '肝硬化 (F=4)',
    'AI Model': 'AI模型',
    'Human CT Report': 'CT报告',
    'Human US Report': '超声报告',
    'Target': '预测靶点',
    'Method': '诊断方法',
    'Sensitivity': '敏感度',
    'Specificity': '特异度',
    'Accuracy': '准确率'
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data():
    try:
        df = pd.read_csv(MASTER_CSV, encoding='utf-8-sig')
    except:
        df = pd.read_csv(MASTER_CSV, encoding='gb18030')
    return df

def get_fusion_cohort(df):
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

def train_and_predict(df, feature_cols, target_col, target_val, comparison='eq'):
    """训练模型并获取预测结果"""
    X = df[feature_cols].values
    
    if comparison == 'ge':
        y = (df[target_col] >= target_val).astype(int).values
    else:
        y = (df[target_col] == target_val).astype(int).values
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.full(len(y), -1)
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        
        # Preprocessing
        imputer = KNNImputer(n_neighbors=5)
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_imp)
        X_test_s = scaler.transform(X_test_imp)
        
        # SMOTE
        k = min(np.sum(y_train) - 1, 5)
        if k > 0:
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_train_s, y_train = smote.fit_resample(X_train_s, y_train)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        clf.fit(X_train_s, y_train)
        
        preds = clf.predict(X_test_s)
        y_pred[test_idx] = preds
    
    return y, y_pred

def calculate_metrics(y_true, y_pred):
    """计算敏感度、特异度、准确率"""
    sens = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = accuracy_score(y_true, y_pred)
    return sens, spec, acc

def parse_human_labels(json_data):
    """解析人类医生的诊断标签"""
    human_labels = {}
    
    for entry in json_data:
        pid = str(entry.get('patient_id', '')).strip()
        if not pid:
            continue
        
        ct_text = str(entry.get('CT', '')).lower()
        us_text = str(entry.get('Bchao', '')).lower()
        
        # S=3 重度脂肪变性
        is_sev_fat = lambda t: ('重度' in t or '中重度' in t) and ('脂肪' in t)
        # S≥2 中重度脂肪变性（包括中度、中重度、重度）
        is_mod_sev_fat = lambda t: ('中度' in t or '中重度' in t or '重度' in t) and ('脂肪' in t)
        # F=4 肝硬化
        is_cirr = lambda t: '肝硬化' in t
        
        human_labels[pid] = {
            'Human_CT_S_Sev': int(is_sev_fat(ct_text)),
            'Human_CT_S_Ge2': int(is_mod_sev_fat(ct_text)),
            'Human_CT_Cirr': int(is_cirr(ct_text)),
            'Human_US_S_Sev': int(is_sev_fat(us_text)),
            'Human_US_S_Ge2': int(is_mod_sev_fat(us_text)),
            'Human_US_Cirr': int(is_cirr(us_text))
        }
    
    return human_labels

def main():
    ensure_dir(OUTPUT_DIR)
    
    # 加载数据
    df = load_data()
    fusion_df = get_fusion_cohort(df)
    fusion_df['patient_id'] = fusion_df['patient_id'].astype(str).str.strip()
    
    logging.info(f"融合队列样本量: {len(fusion_df)}")
    logging.info(f"S≥2阳性样本: {(fusion_df['Steatosis_Grade'] >= 2).sum()}")
    logging.info(f"S=3阳性样本: {(fusion_df['Steatosis_Grade'] == 3).sum()}")
    
    # 加载人类医生标签
    with open(ORIGINAL_JSON, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    human_labels = parse_human_labels(json_data)
    
    h_df = pd.DataFrame.from_dict(human_labels, orient='index').reset_index().rename(columns={'index': 'patient_id'})
    merged_df = pd.merge(fusion_df, h_df, on='patient_id', how='left')
    
    # 获取融合特征
    feature_sets = get_feature_sets(merged_df)
    fusion_feats = feature_sets['Fusion']
    
    # 定义任务
    tasks = [
        {'name': 'S_Ge2', 'col': 'Steatosis_Grade', 'val': 2, 'comparison': 'ge',
         'h_ct': 'Human_CT_S_Ge2', 'h_us': 'Human_US_S_Ge2'},
        {'name': 'S_Sev', 'col': 'Steatosis_Grade', 'val': 3, 'comparison': 'eq',
         'h_ct': 'Human_CT_S_Sev', 'h_us': 'Human_US_S_Sev'},
        {'name': 'F_Cirr', 'col': 'Fibrosis_Stage', 'val': 4, 'comparison': 'eq',
         'h_ct': 'Human_CT_Cirr', 'h_us': 'Human_US_Cirr'}
    ]
    
    rows = []
    
    for task in tasks:
        logging.info(f"处理任务: {task['name']}")
        
        # AI模型预测
        y_true, y_pred_ai = train_and_predict(
            merged_df, fusion_feats, task['col'], task['val'], task['comparison']
        )
        
        if np.any(y_pred_ai == -1):
            logging.error(f"部分样本未被预测: {task['name']}")
            continue
        
        # AI指标
        sens_ai, spec_ai, acc_ai = calculate_metrics(y_true, y_pred_ai)
        rows.append({
            '预测靶点': TRANS_DICT[task['name']],
            '诊断方法': TRANS_DICT['AI Model'],
            '敏感度': f"{sens_ai:.3f}",
            '特异度': f"{spec_ai:.3f}",
            '准确率': f"{acc_ai:.3f}"
        })
        logging.info(f"  AI模型: 敏感度={sens_ai:.3f}, 特异度={spec_ai:.3f}")
        
        # CT报告
        valid_ct = merged_df[merged_df[task['h_ct']].notna()]
        if not valid_ct.empty:
            if task['comparison'] == 'ge':
                y_v = (valid_ct[task['col']] >= task['val']).astype(int)
            else:
                y_v = (valid_ct[task['col']] == task['val']).astype(int)
            y_h = valid_ct[task['h_ct']].astype(int)
            
            sens_h, spec_h, acc_h = calculate_metrics(y_v.values, y_h.values)
            rows.append({
                '预测靶点': TRANS_DICT[task['name']],
                '诊断方法': TRANS_DICT['Human CT Report'],
                '敏感度': f"{sens_h:.3f}",
                '特异度': f"{spec_h:.3f}",
                '准确率': f"{acc_h:.3f}"
            })
            logging.info(f"  CT报告: 敏感度={sens_h:.3f}, 特异度={spec_h:.3f}")
        
        # 超声报告
        valid_us = merged_df[merged_df[task['h_us']].notna()]
        if not valid_us.empty:
            if task['comparison'] == 'ge':
                y_v = (valid_us[task['col']] >= task['val']).astype(int)
            else:
                y_v = (valid_us[task['col']] == task['val']).astype(int)
            y_h = valid_us[task['h_us']].astype(int)
            
            sens_h, spec_h, acc_h = calculate_metrics(y_v.values, y_h.values)
            rows.append({
                '预测靶点': TRANS_DICT[task['name']],
                '诊断方法': TRANS_DICT['Human US Report'],
                '敏感度': f"{sens_h:.3f}",
                '特异度': f"{spec_h:.3f}",
                '准确率': f"{acc_h:.3f}"
            })
            logging.info(f"  超声报告: 敏感度={sens_h:.3f}, 特异度={spec_h:.3f}")
    
    # 创建DataFrame并保存
    result_df = pd.DataFrame(rows)
    output_path = os.path.join(OUTPUT_DIR, "表7.csv")
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    logging.info(f"表格已保存: {output_path}")
    print("\n表格预览:")
    print(result_df.to_string(index=False))

if __name__ == "__main__":
    main()
