# MASLD Opportunistic CT Screening

基于机会性CT筛查的MASLD多模态定量分期模型构建及临床应用价值研究

本仓库为上述大论文的代码仓库。

## 环境配置

### 1. 创建 Conda 环境

```bash
conda create -n paper0 python=3.11 -y
conda activate paper0
```

### 2. 安装依赖

```bash
# 核心科学计算
pip install numpy pandas scipy matplotlib seaborn

# 机器学习
pip install scikit-learn imbalanced-learn xgboost lightgbm catboost shap

# 医学影像处理
pip install pydicom SimpleITK nibabel dicom2nifti

# 影像组学
pip install pyradiomics

# 体成分分割 (TotalSegmentator)
pip install totalsegmentator

# 其他
pip install Pillow openpyxl
```

或使用 GPU 加速版本的 TotalSegmentator：

```bash
pip install totalsegmentator[gpu]
```

### 3. 中文字体配置（Windows）

确保系统安装 `SimHei`（黑体）字体，路径为 `C:\Windows\Fonts\simhei.ttf`。代码中 `task_18_plot_CN.py` 会通过 matplotlib 的 `font_manager` 自动加载。

### 4. 目录结构

```
paper0/
├── code/               # 全部 Python 代码（38个脚本）
├── output/             # 输出目录（表格、图片）
│   ├── master_dataset_final.csv
│   ├── model_arena_leaderboard.csv
│   └── thesis_assets/
│       ├── tables/
│       └── figures/
└── data/               # 原始数据
```

### 5. 运行

按任务编号顺序执行 `code/` 目录下的脚本：

```bash
conda activate paper0
cd code
python task_1_ct_inventory.py
python task_4_feature_extractor.py
# ... 按顺序执行到 task_18_plot_CN.py
```

核心脚本说明：
- `task_4_feature_extractor.py` — PyRadiomics 影像组学特征提取
- `task_8_organ_segmentator.py` — TotalSegmentator 体成分分割
- `task_12_model_arena.py` — 多模型训练与评估
- `task_17.1_table_factory.py` — 论文全部表格生成
- `task_18_plot_CN.py` — 中文版图表生成
