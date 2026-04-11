# MASLD Opportunistic CT Screening

基于胸部CT机会性筛查的MASLD多模态定量分期模型构建及临床应用价值研究

## 项目概述

本项目利用常规胸部CT检查的机会性筛查数据，结合影像组学特征、体成分分析和临床指标，构建代谢相关脂肪性肝病（MASLD）的多模态定量分期模型。

## 核心代码说明

| 文件 | 功能描述 |
|------|----------|
| `task_1_ct_inventory.py` | CT影像数据清点与元数据提取 |
| `task_4_feature_extractor.py` | PyRadiomics影像组学特征提取 |
| `task_6_index_calculator.py` | FIB-4、APRI等临床指数计算 |
| `task_8_organ_segmentator.py` | TotalSegmentator体成分分割（肝脏、脾脏、肌肉） |
| `task_8_body_supplement.py` | 体成分特征补充计算 |
| `task_10_final_modeling.py` | 最终预测模型训练与评估 |
| `task_12_model_arena.py` | 多模型竞技场（LR/SVM/RF/XGB/LGBM/CAT） |
| `task_17.1_table_factory.py` | 论文表格生成 |
| `task_17.3_plot_modeling.py` | 模型性能可视化（ROC、校准曲线、DCA） |
| `task_17.4_plot_validation.py` | 模型验证可视化（SHAP、人类vs AI对比） |
| `task_18_plot_CN.py` | 中文版图表生成 |

## 技术栈

- **深度学习分割**: TotalSegmentator
- **影像组学**: PyRadiomics
- **机器学习**: scikit-learn, XGBoost, LightGBM, CatBoost
- **不平衡处理**: SMOTE (imbalanced-learn)
- **可解释性**: SHAP
- **可视化**: matplotlib, seaborn

## 主要功能

### 1. 体成分自动分割
```python
# 使用TotalSegmentator进行器官分割
ts_api.totalsegmentator(
    ct_path, seg_out_dir,
    roi_subset=['liver', 'spleen'],
    fast=True, device="gpu:0"
)
```

### 2. 影像组学特征提取
```python
# PyRadiomics特征提取
extractor = featureextractor.RadiomicsFeatureExtractor()
features = extractor.execute(ct_path, mask_path)
```

### 3. 多模型训练与评估
```python
# SMOTE + Pipeline
pipe = ImbPipeline([
    ('imputer', KNNImputer()),
    ('scaler', StandardScaler()),
    ('smote', SMOTE()),
    ('clf', RandomForestClassifier())
])
```

### 4. SHAP可解释性分析
```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)
```

## 研究结果

| 预测靶点 | AI模型敏感度 | CT报告敏感度 |
|----------|-------------|-------------|
| 中重度脂肪变性 (S≥2) | 84.8% | 0% |
| 重度脂肪变性 (S=3) | 73.8% | 0% |

**关键发现**: AI模型在中重度脂肪变性诊断上显著优于人类医生，解决了机会性筛查中漏诊率高的问题。

## 环境依赖

```
python >= 3.8
totalsegmentator
pyradiomics
scikit-learn
xgboost
lightgbm
catboost
imbalanced-learn
shap
nibabel
pandas
numpy
matplotlib
seaborn
```

## 引用

如果本项目对您的研究有帮助，请引用：

```
@article{masld_ct_screening_2026,
  title={基于胸部CT机会性筛查的MASLD多模态定量分期模型构建及临床应用价值研究},
  author={Your Name},
  year={2026}
}
```

## 许可证

MIT License
