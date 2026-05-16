import os
import pandas as pd
import logging
import datetime

# --- Configuration ---
BASE_DIR = "/mnt/d/mWork/paper0"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FINAL_DOC = os.path.join(OUTPUT_DIR, "thesis_material_compendium.md")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_md(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return f"> *Warning: Report {filename} not found.*"
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    lines = []
    lines.append(f"# 临床专硕学位论文素材汇编 (Thesis Compendium)")
    lines.append(f"**生成时间**: {datetime.datetime.now()}")
    lines.append(f"**项目代号**: Paper0 (MAFLD 多模态影像组学研究)")
    lines.append("")
    lines.append("> **文档说明**：本文档旨在为 8 万字大论文提供全方位的素材支持，涵盖数据治理、特征工程、模型迭代及临床验证的全过程。重点记录“踩坑”与“填坑”的工程细节。")
    lines.append("")

    # --- Chapter 1: 数据治理 ---
    lines.append("# 第一章：数据治理与质量控制 (Data Governance)")
    lines.append("## 1.1 原始数据的混沌状态")
    lines.append("在研究初期，我们面临着极度复杂的 DICOM 数据环境。不同的扫描协议、重建核和层厚混合在一起。")
    lines.append("### 证据素材 (From Task 1 & 2)")
    lines.append(read_md("task_1_analysis_report_v2.md"))
    lines.append(read_md("task_2_run_report.md"))
    
    lines.append("## 1.2 物理清洗与 ID 修复")
    lines.append("数据清洗不仅仅是筛选，更是物理层面的重组。我们遇到了文件路径过长、硬链接创建失败等系统级问题，最终通过 WSL2 环境解决。")
    lines.append("### 证据素材 (From Task 3 & 5.6)")
    lines.append(read_md("task_3_run_report.md"))
    # 手动补充关于 ID 修复的描述
    lines.append("### 关键技术难点：ID 链路断裂与修复")
    lines.append("在 Task 5 中，我们发现 `cleaned_series` 数据库仅记录了 60 例患者，而物理文件有 165 例。更严重的是，清洗后的文件夹名变成了无意义的 Accession Number。")
    lines.append("**解决方案**：我们开发了基于 MFT (Master File Table) 硬链接追踪技术 (Task 5.6)，逆向解析出每个 DICOM 文件的原始路径，成功找回了 165 例患者的真实住院号，实现了 100% 的 ID 还原。这是保证多模态数据对齐的关键一步。")

    # --- Chapter 2: 特征工程 ---
    lines.append("\n# 第二章：多模态特征工程 (Multimodal Feature Engineering)")
    lines.append("## 2.1 影像组学特征 (Radiomics)")
    lines.append("基于 PyRadiomics 提取了高维纹理特征。")
    lines.append("### 证据素材 (From Task 4)")
    lines.append(read_md("task_4_supplement_report_final.md"))
    
    lines.append("## 2.2 宏观体成分分析 (Body Composition)")
    lines.append("这是本研究的创新核心。我们不满足于微观纹理，进一步提取了具有明确生理意义的器官体积与密度。")
    lines.append("### 证据素材 (From Task 8)")
    lines.append(read_md("task_8_report.md"))
    lines.append("### 技术攻坚：TotalSegmentator 的适配")
    lines.append("在提取肌肉和脂肪时，我们遭遇了严重的版本兼容性问题（Task 8.2）。")
    lines.append("*   **问题**: `fast=True` 模式不支持肌肉分割；v2 版 API 对 `roi_subset` 支持不佳。")
    lines.append("*   **解决**: 申请官方学术授权，转用 CLI 命令行模式调用 `tissue_types` 任务，成功提取了 VAT (内脏脂肪) 和 SAT (皮下脂肪)。")

    # --- Chapter 3: 模型演进 ---
    lines.append("\n# 第三章：模型迭代与优化 (Model Evolution)")
    lines.append("## 3.1 初步尝试：影像组学的局限")
    lines.append("最初的实验表明，仅靠 CT 纹理特征预测早期纤维化效果不佳。")
    lines.append("### 证据素材 (From Task 7)")
    lines.append(read_md("task_7_run_report.md"))
    
    lines.append("## 3.2 策略转型：全靶点搜索")
    lines.append("既然早期纤维化难预测，我们转换思路，全面扫描了脂肪变、炎症和晚期肝硬化四个靶点。")
    lines.append("### 证据素材 (From Task 11)")
    lines.append(read_md("task_11_run_report.md"))
    
    lines.append("## 3.3 终极竞技：算法与采样策略")
    lines.append("为了解决 F=4 样本极少 (N=12) 的问题，我们引入了 SMOTE 过采样，并对比了 7 种机器学习模型。")
    lines.append("### 证据素材 (From Task 12)")
    lines.append(read_md("task_12_run_report.md"))

    # --- Chapter 4: 临床价值 ---
    lines.append("\n# 第四章：可解释性与临床价值 (Clinical Translation)")
    lines.append("## 4.1 机器是如何思考的？(SHAP)")
    lines.append("我们解构了冠军模型（RandomForest），发现了 `Muscle_Mean_HU` 在肝硬化预测中的重要性。")
    lines.append("### 证据素材 (From Task 13)")
    lines.append(read_md("task_13_run_report.md"))
    
    lines.append("## 4.2 人机大战 (Human vs AI)")
    lines.append("最激动人心的环节：AI 在重度脂肪肝诊断上击败了人类医生的文本报告。")
    lines.append("### 证据素材 (From Task 14)")
    lines.append(read_md("task_14_run_report.md"))

    # --- Appendix: 避坑指南 ---
    lines.append("\n# 附录：工程难点与解决方案汇总 (Pitfalls & Solutions)")
    lines.append("在 8 万字的论文中，这一部分可以作为“材料与方法”或“讨论”章节的补充，体现工作量。")
    lines.append("1.  **环境迁移**: 从 Windows 到 WSL2 的迁移是必须的，因为 `dicom2nifti` 和 `TotalSegmentator` 在 Linux 下表现更稳定，且 GPU 调用更顺畅。")
    lines.append("2.  **DICOM 几何失真**: 也就是 Task 4.5 中遇到的 `Slice increment not consistent`。我们保留了详细的错误日志，证明了我们对数据质量的严苛要求（剔除 1 例）。")
    lines.append("3.  **多模态对齐**: 临床数据（CSV）与影像数据（Folder）的 ID 格式不一致是医学大数据的通病。我们编写了专门的 ID 映射脚本来解决此问题。")
    
    # Save
    with open(FINAL_DOC, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    logging.info(f"Compendium saved to {FINAL_DOC}")

if __name__ == "__main__":
    main()
