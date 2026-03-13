# 科研龙虾·Research-Claw

> 医学影像 AI 科研全流程工具与方法论指南

---

## 📋 目录

1. [项目简介](#项目简介)
2. [科研流程总览](#科研流程总览)
3. [文献调研工具](#文献调研工具)
4. [数据获取与预处理](#数据获取与预处理)
5. [实验设计与基线复现](#实验设计与基线复现)
6. [模型训练与调优](#模型训练与调优)
7. [评估指标与可视化](#评估指标与可视化)
8. [论文写作与投稿](#论文写作与投稿)
9. [开源发布检查清单](#开源发布检查清单)
10. [常见踩坑与解决方案](#常见踩坑与解决方案)

---

## 🦀 项目简介

**科研龙虾（Research-Claw）** 是一套面向医学影像 AI 研究者的系统化科研方法论框架。如同龙虾以强健的螯足精准捕捉猎物，Research-Claw 帮助研究者以体系化的方式抓住每个科研关键节点，避免常见错误，高效推进从想法到论文发表的全流程。

**适用人群：**
- 医学影像 AI 方向的研究生（硕士/博士）
- 希望进入医学 AI 领域的工程师
- 开展 AI 辅助诊断研究的临床医生

**核心理念：**
- 🎯 **精准发力**：在关键节点投入足够精力，避免无效迭代
- 🔄 **持续迭代**：小步快跑，快速验证假设
- 📊 **数据驱动**：用数据说话，而非直觉
- 🤝 **跨学科合作**：医工结合，临床需求驱动技术创新

---

## 🔄 科研流程总览

```
[问题发现] → [文献调研] → [数据准备] → [方法设计] → [实验验证] → [论文写作] → [开源发布]
     ↑                                                                              |
     └──────────────────────── 持续迭代反馈 ──────────────────────────────────────┘
```

### 各阶段时间分配参考

| 阶段 | 典型时长 | 关键产出 | 常见陷阱 |
|------|---------|---------|---------|
| 问题发现与调研 | 2–4 周 | 研究方向、Gap 分析 | 调研不充分导致重复研究 |
| 数据准备 | 2–8 周 | 清洗后数据集、标注指南 | 数据泄漏、标注不一致 |
| 方法设计与实现 | 4–12 周 | 可运行代码、基线结果 | 过度工程化、忽略基线 |
| 实验与消融 | 4–8 周 | 完整实验记录、关键结论 | 缺乏消融实验、p-hacking |
| 论文写作 | 2–6 周 | 初稿、修改稿 | 图表质量差、贡献不清晰 |
| 审稿与修改 | 视情况 | 接收通知 | 回复审稿人不充分 |

---

## 📚 文献调研工具

### 必备工具清单

| 工具 | 用途 | 推荐指数 | 链接 |
|------|------|---------|------|
| **PubMed** | 医学文献检索 | ⭐⭐⭐⭐⭐ | [pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov/) |
| **Google Scholar** | 广泛学术检索 | ⭐⭐⭐⭐⭐ | [scholar.google.com](https://scholar.google.com/) |
| **arXiv** | 预印本，追踪最新进展 | ⭐⭐⭐⭐⭐ | [arxiv.org](https://arxiv.org/) |
| **Papers with Code** | 代码 + 论文 + 排行榜 | ⭐⭐⭐⭐⭐ | [paperswithcode.com](https://paperswithcode.com/) |
| **Semantic Scholar** | AI 辅助文献分析 | ⭐⭐⭐⭐ | [semanticscholar.org](https://www.semanticscholar.org/) |
| **Connected Papers** | 文献关系图谱 | ⭐⭐⭐⭐ | [connectedpapers.com](https://www.connectedpapers.com/) |
| **Zotero** | 文献管理 | ⭐⭐⭐⭐⭐ | [zotero.org](https://www.zotero.org/) |
| **Notion / Obsidian** | 笔记与知识管理 | ⭐⭐⭐⭐ | — |

### 高效调研策略

#### 第一轮：快速全局扫描（1–3 天）
1. 在 PubMed 和 Google Scholar 检索核心关键词，收集近 3 年综述论文
2. 阅读 2–3 篇综述，建立领域知识框架
3. 识别该方向的顶会/顶刊（如 MICCAI、TMI、Radiology）
4. 在 Papers with Code 查找该任务的 SOTA 排行榜

#### 第二轮：深度精读（1–2 周）
1. 精读 10–20 篇核心论文，重点关注方法、数据、指标
2. 使用 Connected Papers 找出被高频引用的奠基性工作
3. 建立论文对比表格（见下方模板）
4. 明确 **Research Gap**：现有方法在哪些场景/数据集/指标上表现不足？

#### 论文对比表格模板

| 论文 | 年份 | 方法核心 | 数据集 | 指标 | 局限性 | 我的机会点 |
|------|------|---------|-------|------|-------|---------|
| CheXNet | 2017 | DenseNet121 | ChestX-ray14 | AUC 0.841 | 单一数据集 | 多中心泛化 |
| MedSAM | 2023 | SAM 微调 | 11 个模态 | Dice 0.876 | 交互式分割 | 全自动化 |
| … | … | … | … | … | … | … |

---

## 💾 数据获取与预处理

### 公开数据集资源

参考主文档 [medical-imaging-ai-models-2025.md](./medical-imaging-ai-models-2025.md) 中的"重要数据集"章节。

### 数据申请流程

```
1. 注册账号（PhysioNet / Kaggle / 机构官网）
2. 完成数据使用协议（DUA）签署
3. 提交研究用途说明（IRB 批准号、机构背书）
4. 等待审批（通常 1–4 周）
5. 下载数据并验证完整性（MD5 校验）
```

### 数据预处理标准流程

```python
# 医学影像预处理通用流程示例
import numpy as np
import SimpleITK as sitk

def preprocess_ct(input_path: str, output_path: str,
                  hu_min: float = -1000, hu_max: float = 400,
                  target_spacing: tuple = (1.0, 1.0, 1.0)):
    """
    CT 影像标准化预处理：
    1. 重采样到目标分辨率
    2. HU 值窗宽窗位裁剪
    3. 归一化到 [0, 1]
    """
    image = sitk.ReadImage(input_path)

    # 重采样
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(orig * orig_sp / tgt_sp))
        for orig, orig_sp, tgt_sp in zip(original_size, original_spacing, target_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    image = resampler.Execute(image)

    # HU 裁剪与归一化
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    array = np.clip(array, hu_min, hu_max)
    array = (array - hu_min) / (hu_max - hu_min)

    result = sitk.GetImageFromArray(array)
    result.CopyInformation(image)
    sitk.WriteImage(result, output_path)
```

### 数据质量检查清单

- [ ] 检查图像维度和分辨率分布
- [ ] 统计标签分布，确认类别不平衡程度
- [ ] 可视化随机样本（至少 50 张），人工抽查质量
- [ ] 确认训练/验证/测试集无患者级别泄漏（按患者 ID 而非图像 ID 划分）
- [ ] 对比数据集统计信息与论文描述是否一致
- [ ] 检查 DICOM 元数据（设备型号、扫描参数）分布

### 数据标注规范

| 项目 | 建议 |
|------|------|
| 标注工具 | ITK-SNAP（分割）、LabelImg（检测）、CVAT（通用） |
| 标注人员 | ≥2 名标注员独立标注，计算 Inter-rater agreement（Kappa/ICC） |
| 标注指南 | 书面记录，明确边界处理规则 |
| 质控流程 | 10% 随机抽查，不合格返工 |
| 版本管理 | 使用 DVC 管理数据版本 |

---

## 🧪 实验设计与基线复现

### 实验设计原则

1. **先跑基线，再搞创新**：确保你能复现 SOTA，再尝试改进
2. **控制变量**：每次只改一个因素
3. **统计显著性**：3 次以上独立实验取均值±标准差
4. **消融实验**：逐一验证每个改进的贡献

### 基线复现检查清单

- [ ] 代码来源（官方实现 > 第三方复现 > 自行实现）
- [ ] 使用与原论文相同的数据集版本
- [ ] 使用相同的数据划分（或联系作者获取）
- [ ] 超参数与论文一致（学习率、batch size、epoch 数）
- [ ] 结果误差在原论文 ±1% 以内（否则需查找原因）

### 实验追踪工具

| 工具 | 特点 | 适用场景 |
|------|------|---------|
| **MLflow** | 开源、自托管 | 团队/机构内部 |
| **Weights & Biases** | 功能强大、可视化好 | 个人/小团队 |
| **TensorBoard** | PyTorch/TF 内置 | 快速查看训练曲线 |
| **DVC** | 数据版本 + 实验追踪 | 大规模数据管理 |

### 标准实验记录模板

```yaml
# experiment_config.yaml
experiment_id: "exp_001_medsam_chestxray"
date: "2025-03-01"
author: "Research-Claw Team"

dataset:
  name: "ChestX-ray14"
  version: "v1.0"
  split: "official_split_v2"
  train_samples: 86524
  val_samples: 25596
  test_samples: 25596

model:
  architecture: "MedSAM"
  pretrained_weights: "medsam_vit_b.pth"
  input_size: [1024, 1024]

training:
  optimizer: "AdamW"
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  epochs: 50
  batch_size: 4
  hardware: "2x A100 80GB"

results:
  val_auc: 0.873
  test_auc: 0.869
  inference_time_ms: 42
  gpu_memory_gb: 12.3

notes: "首次实验，使用默认超参数"
```

---

## 🔧 模型训练与调优

### 训练稳定性检查清单

- [ ] 损失函数在前几个 epoch 稳定下降
- [ ] 无 NaN/Inf 损失（检查数值稳定性）
- [ ] 学习率调度曲线符合预期
- [ ] 验证集指标与训练集指标无明显 gap（检查过拟合）
- [ ] GPU 利用率 > 70%（检查数据加载瓶颈）

### 常用超参数调优策略

```python
# 学习率范围测试（Learning Rate Finder）
# 在正式训练前，快速扫描最佳学习率范围

from torch.optim.lr_scheduler import OneCycleLR

# 推荐起点配置（医学影像分类任务）
config = {
    "lr": 1e-4,           # 预训练模型微调
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "max_epochs": 100,
    "scheduler": "cosine_with_warmup",
}

# 对于分割任务（MedSAM / nnU-Net 风格）
seg_config = {
    "lr": 1e-4,
    "poly_lr_decay": True,  # 多项式衰减
    "loss": "DiceCE",       # Dice + CrossEntropy
    "deep_supervision": True,
}
```

### 医学影像常用数据增强

```python
import albumentations as A

# 分类任务增强
classification_transforms = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2, contrast_limit=0.2, p=0.5
    ),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.Normalize(mean=[0.485], std=[0.229]),  # 单通道 X-ray
])

# 分割任务增强（保持 image-mask 一致性）
segmentation_transforms = A.Compose([
    A.RandomCrop(width=512, height=512),
    A.HorizontalFlip(p=0.5),
    A.ElasticTransform(alpha=120, sigma=12, p=0.3),
    A.GridDistortion(p=0.2),
    A.Normalize(),
])
```

---

## 📊 评估指标与可视化

### 医学影像 AI 常用评估指标

#### 分类任务

| 指标 | 公式 | 适用场景 | 注意事项 |
|------|------|---------|---------|
| AUC-ROC | 见下 | 二分类，不平衡数据 | 需报告 95% 置信区间 |
| F1 Score | 2TP/(2TP+FP+FN) | 不平衡数据 | 指定阈值 |
| Sensitivity | TP/(TP+FN) | 筛查场景 | 同时报告 Specificity |
| Specificity | TN/(TN+FP) | 降低假阳性 | 同时报告 Sensitivity |
| AURPC | — | 极端不平衡 | 优于 AUC-ROC |

#### 分割任务

| 指标 | 公式 | 适用场景 |
|------|------|---------|
| Dice 系数 | 2\|X∩Y\|/(|X|+|Y|) | 通用分割评估 |
| IoU / Jaccard | \|X∩Y\|/\|X∪Y\| | 目标检测/分割 |
| HD95 | 95th percentile HD | 边界精度评估 |
| NSD | — | 表面距离评估 |

#### 报告生成任务

| 指标 | 工具 | 注意事项 |
|------|------|---------|
| BLEU-1/4 | NLTK | 需与临床相关性结合 |
| ROUGE-L | rouge-score | 召回导向 |
| CIDEr | — | 专为图像描述设计 |
| BERTScore | transformers | 语义相似度 |
| CheXBert F1 | CheXBert | 临床标签一致性（推荐） |

### 结果可视化标准

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc

def plot_roc_with_ci(y_true, y_scores, n_bootstrap=1000, figsize=(6, 6)):
    """绘制带 95% 置信区间的 ROC 曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Bootstrap 置信区间
    aucs = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        f, t, _ = roc_curve(y_true[idx], y_scores[idx])
        aucs.append(auc(f, t))

    ci_lower, ci_upper = np.percentile(aucs, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color='#E63946', lw=2,
            label=f'AUC = {roc_auc:.3f} [{ci_lower:.3f}–{ci_upper:.3f}]')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    return fig
```

---

## ✍️ 论文写作与投稿

### 目标会议/期刊推荐

#### 顶会（CCF-A / Q1）

| 会议/期刊 | 截稿时间（参考） | 录用率 | 特点 |
|---------|--------------|-------|------|
| **MICCAI** | 3 月初 | ~30% | 医学影像顶会，口头/海报 |
| **CVPR** | 11 月中 | ~26% | CV 顶会，接受医学影像 |
| **NeurIPS** | 5 月 | ~26% | AI 顶会，影响力大 |
| **IEEE TMI** | 随时 | ~30% | 医学影像顶刊，影响因子 >10 |
| **Radiology: AI** | 随时 | — | 临床影响力强 |
| **Medical Image Analysis** | 随时 | ~25% | 纯医学影像期刊 |

#### 投稿建议

1. **MICCAI 首选**：医学影像方向最重要的会议
2. **先会议后期刊**：会议版本 + 扩展期刊版本（MICCAI → TMI）
3. **arXiv 预印本**：接受后立即上传，提高曝光度
4. **开源代码**：接受率和引用量均更高

### 论文结构参考

```
1. Abstract（250 词以内）
   - 问题 → 方法 → 结果 → 结论

2. Introduction
   - 临床动机（为什么这个问题重要？）
   - 现有方法局限性（Research Gap）
   - 本文贡献（Contributions，3–4 条）

3. Related Work
   - 相关任务综述
   - 与本文的区别

4. Method
   - 问题定义（数学符号）
   - 网络结构图（必须有！）
   - 各模块详细描述
   - 损失函数

5. Experiments
   - 数据集与评估指标
   - 实现细节（可复现性）
   - 与 SOTA 对比（主实验）
   - 消融实验
   - 定性可视化

6. Conclusion
   - 总结贡献
   - 局限性与未来工作

7. References
```

### 写作建议

- **图表为王**：一张清晰的方法图胜过千言万语
- **数字要精确**：AUC 0.873，而非"我们的方法表现更好"
- **消融实验完整**：每个创新点都需要有对应的消融验证
- **代码和数据可用性声明**：提升复现性，加分项

---

## 📦 开源发布检查清单

### 代码发布前检查

- [ ] **README** 包含：安装、快速开始、数据集说明、训练/推理命令
- [ ] **requirements.txt** 或 **environment.yaml** 版本锁定
- [ ] **预训练权重** 上传至 Hugging Face 或 Google Drive（附 MD5）
- [ ] **数据预处理脚本** 包含完整复现流程
- [ ] **示例 notebook** 演示关键功能
- [ ] **许可证**（医学代码推荐 Apache 2.0 或 MIT）
- [ ] **引用信息**（BibTeX 格式）

### 标准 README 结构

```markdown
# 项目名称

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-red)](链接)
[![GitHub Stars](https://img.shields.io/github/stars/xxx/xxx)](链接)

## 简介
...

## 安装
pip install -r requirements.txt

## 快速开始
python inference.py --input path/to/image.dcm --output result.png

## 数据准备
...

## 训练
...

## 引用
@inproceedings{xxx2025,
  title={...},
  author={...},
  booktitle={MICCAI},
  year={2025}
}
```

---

## ⚠️ 常见踩坑与解决方案

### 数据相关

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 测试集结果远低于验证集 | 数据泄漏 | 按患者 ID 划分，而非图像 ID |
| 不同数据集结果差异大 | 域偏移 | 多中心验证，TTA（测试时增强） |
| 训练收敛后性能突然下降 | 学习率过大 | 使用 warmup + 余弦退火 |
| 类别不平衡导致模型偏向大类 | 标签分布不均 | 类别权重、过采样、Focal Loss |

### 实验相关

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 无法复现论文结果 | 超参数/数据版本差异 | 联系作者，检查数据划分 |
| 消融实验结论矛盾 | 随机性 | 多次实验取均值，固定随机种子 |
| 模型过拟合 | 数据量不足 | 数据增强、预训练初始化、Dropout |
| 推理速度过慢 | 模型过大 | 知识蒸馏、量化、TensorRT 加速 |

### 写作与审稿相关

| 问题 | 解决方案 |
|------|---------|
| 审稿人要求更多对比实验 | 预先在实验设计阶段留出 buffer |
| 图表质量差被拒 | 统一使用矢量图（PDF/SVG），字号 ≥ 10pt |
| 贡献不清晰 | Introduction 明确列出 3–4 条 Contributions |
| 方法描述不可复现 | 完整报告超参数，提供开源代码 |

---

## 🔗 延伸资源

### 必读综述

1. **医学影像 AI 入门**：Litjens et al. (2017). "A survey on deep learning in medical image analysis." *Medical Image Analysis*
2. **Transformer in Medical Imaging**：Shamshad et al. (2023). "Transformers in medical imaging: A survey." *Medical Image Analysis*
3. **基础模型 + 医学**：Moor et al. (2023). "Foundation models for generalist medical artificial intelligence." *Nature*

### 学习路径

```
入门阶段（1–3 个月）
├── 深度学习基础：PyTorch 官方教程
├── 医学影像基础：MONAI 教程
└── 论文精读：CheXNet, U-Net, nnU-Net

进阶阶段（3–6 个月）
├── 复现一篇 MICCAI 论文
├── 参加 Kaggle 医学影像竞赛
└── 提交第一篇 MICCAI/MIDL 论文

深入阶段（6 个月+）
├── 与临床医生合作设计新问题
├── 构建自己的数据集
└── 发表顶会/顶刊论文
```

### 社区资源

- **MICCAI 官方教程**: https://miccai.org/
- **MIDL 社区**: https://midl.io/
- **Medical Open Network for AI (MONAI)**: https://monai.io/
- **Papers with Code - Medical**: https://paperswithcode.com/area/medical

---

## 📧 贡献与反馈

欢迎通过 Issue 或 Pull Request 提交：
- 新的工具推荐
- 踩坑经验分享
- 错误修正

---

**版本**: v1.0  
**最后更新**: 2025 年 3 月  
**维护者**: 科研龙虾·Research-Claw 项目组
