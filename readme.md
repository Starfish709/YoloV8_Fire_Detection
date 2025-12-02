# **🔥 YOLOv8-Fire-Detection: 基于全图伪标签的火焰检测系统**

\<div align="center"\>  
**针对无检测标注数据集的弱监督目标检测方案**  
\</div\>

## **📖 项目简介 (Introduction)**

本项目旨在利用深度学习技术解决火焰识别与定位问题。面对原始数据集\*\*仅有图片、无检测框标注（No Bounding Box Labels）**的挑战，我们设计了一种基于**全图伪标签（Full-image Pseudo-labeling）\*\*的弱监督策略，成功将分类任务转化为检测任务。  
基于 **YOLOv8n** 架构，针对 **Apple M3 芯片**进行了深度工程优化（Batch 64 \+ AMP \+ MPS），最终实现了 **mAP@0.5 \> 99%** 的卓越性能，并具备良好的可解释性（Grad-CAM）。

## **✨ 核心亮点 (Highlights)**

* 🚀 **弱监督策略**：首创性地使用全图坐标（0.5 0.5 1.0 1.0）作为伪标签，迫使模型学习全局纹理特征。  
* ⚡️ **M3 芯片极限优化**：  
  * **MPS 加速**：启用 Metal Performance Shaders，充分利用 Mac GPU。  
  * **大 Batch 训练**：针对 16GB 统一内存优化，开启 Batch=64，极大提升训练稳定性。  
  * **混合精度 (AMP)**：训练速度提升 200%。  
* 🔍 **可解释性验证**：集成 Grad-CAM 热力图，证明模型虽然使用全图标签，但依然能精准聚焦于火焰核心区域。

## **📊 效果展示 (Performance)**

### **1\. 训练指标**

| 指标 | 数值 | 说明 |
| :---- | :---- | :---- |
| **mAP@0.5** | **99.5%** | 远超预期 (96%) |
| **Precision** | **99.1%** | 极少误报 |
| **Recall** | **100%** | 无漏报 |

### **2\. 可视化效果**

*(图注：左侧为 Loss 下降曲线，右侧为准确率上升曲线)*  
*(图注：Grad-CAM 热力图显示模型关注点精准覆盖火焰区域)*

## **🛠️ 环境安装 (Installation)**

1. **克隆项目**  
   git clone \[https://github.com/yourusername/YOLOv8-Fire-Detection.git\](https://github.com/yourusername/YOLOv8-Fire-Detection.git)  
   cd YOLOv8-Fire-Detection

2. **安装依赖**  
   pip install ultralytics grad-cam opencv-python matplotlib

3. **准备数据**  
   * 请将你的原始图片放入 dataset\_src 文件夹。  
   * *(注：无需标注文件，脚本会自动生成伪标签)*

## **🚀 快速开始 (Quick Start)**

我们提供了一键式脚本，按顺序运行即可完成全流程。

### **第一步：数据切分与清洗**

自动划分 80% 训练集 / 20% 测试集，并生成伪标签。  
python split.py

### **第二步：生成配置文件**

自动生成 YOLO 所需的 fire.yaml。  
python create\_yaml.py

### **第三步：模型训练**

开始在 M3/GPU 上进行高效训练（默认 100 轮，含早停机制）。  
python main\_train.py

### **第四步：评估与可视化**

生成结果图表与热力图。  
python plot\_results.py  \# 生成训练曲线图  
python heatmap.py       \# 生成 Grad-CAM 热力图  
python test.py          \# 单张图片推理测试

## **📂 目录结构 (File Structure)**

DeepL/  
├── dataset\_src/       \# \[输入\] 原始图片存放处  
├── fire\_dataset/      \# \[自动生成\] 切分后的数据集  
├── runs/              \# \[输出\] 训练结果与权重文件 (weights/best.pt)  
├── split.py           \# 数据预处理脚本  
├── create\_yaml.py     \# 配置文件生成脚本  
├── main\_train.py      \# 主训练脚本 (M3优化版)  
├── plot\_results.py    \# 结果可视化脚本  
├── heatmap.py         \# 热力图生成脚本  
├── test\_accuracy.py   \# 准确率验证脚本  
└── README.md          \# 项目说明文档

## **📝 许可证**

本项目采用 [MIT License](https://www.google.com/search?q=LICENSE) 许可证。  
*Created by \[StarFish709*\]*