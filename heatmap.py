import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import random
from pathlib import Path
import sys

# --- 1. 自动定位路径 ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'fire_dataset/images/test'

# 自动寻找最新的训练权重
possible_weights = [
    BASE_DIR / 'runs/fire/yolov8n_fire_m3_b64/weights/best.pt',
    BASE_DIR / 'runs/fire/yolov8n_fire_m3/weights/best.pt',
    BASE_DIR / 'runs/fire/yolov8n_fire_v2/weights/best.pt',
    BASE_DIR / 'runs/fire/yolov8n_fire_extend/weights/best.pt'
]

WEIGHT_PATH = None
for p in possible_weights:
    if p.exists():
        WEIGHT_PATH = p
        break

SAVE_PATH = BASE_DIR / 'heatmap.png'

# --- 尝试导入 Grad-CAM ---
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("!!! 错误: 缺少 'grad-cam' 库")
    print("请运行: pip install grad-cam")
    sys.exit(1)

# --- 2. 包装器 (解决 tuple 报错) ---
class YOLOGradCAMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        result = self.model(x)
        # YOLOv8 推理返回 (output, hidden_states)，我们只要 output
        if isinstance(result, (tuple, list)):
            return result[0]
        return result

# --- 3. 自定义目标函数 (核心修复: 解决 scalar outputs 报错) ---
class YOLOBoxScoreTarget:
    """
    将 YOLO 输出的 [Batch, 4+NC, Anchors] 张量聚合为一个标量分数。
    Grad-CAM 需要一个标量来计算梯度。
    """
    def __init__(self, class_idx=0):
        self.class_idx = class_idx

    def __call__(self, model_output):
        # model_output 形状: [1, 5, 8400] (对于单类别检测)
        # 0-3 是坐标，4 是类别置信度
        if model_output.ndim == 3:
            # 提取所有锚框对 "fire" (class_idx=0) 的预测分数并求和
            # 这代表了模型对整张图“有火”的总信心
            fire_scores = model_output[0, 4 + self.class_idx, :]
            return fire_scores.sum()
        else:
            return model_output.sum()

def get_target_layer(model):
    """寻找目标层"""
    try:
        # 取 Detect 头之前的最后一层
        target = model.model[-2]
        if hasattr(target, 'cv2'):
            return [target.cv2]
        return [target]
    except Exception as e:
        print(f"警告: 自动寻找目标层失败: {e}")
        return [list(model.model.children())[-2]]

def main():
    if not WEIGHT_PATH:
        print("!!! 错误: 找不到训练好的模型文件。")
        return

    print(f">>> 加载模型: {WEIGHT_PATH}")
    yolo_model = YOLO(str(WEIGHT_PATH))
    
    # 提取 PyTorch 原生模型
    nn_model = yolo_model.model
    
    # 【修复1】强制开启模型参数梯度
    for param in nn_model.parameters():
        param.requires_grad = True

    nn_model.eval() 

    # 【修复2】套上包装器
    wrapper_model = YOLOGradCAMWrapper(nn_model)

    # 寻找测试图片
    if not DATA_DIR.exists():
        print(f"!!! 错误: 测试集目录不存在")
        return
        
    img_list = list(DATA_DIR.glob('*'))
    if not img_list:
        print("!!! 错误: 测试集里没有图片")
        return

    # 随机选一张图
    img_path = random.choice(img_list)
    print(f">>> 正在处理图片: {img_path.name}")

    img = cv2.imread(str(img_path))
    if img is None: return
    
    img = cv2.resize(img, (640, 640))
    rgb_img = np.float32(img) / 255
    input_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0)

    # 【修复3】强制输入图片梯度
    input_tensor.requires_grad = True

    # 获取目标层
    target_layers = get_target_layer(nn_model)
    
    # 构建 GradCAM
    cam = GradCAM(model=wrapper_model, target_layers=target_layers)
    
    # 【修复4】使用自定义目标函数
    targets = [YOLOBoxScoreTarget(class_idx=0)]
    
    try:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    except Exception as e:
        print(f"!!! 计算出错: {e}")
        return

    grayscale_cam = grayscale_cam[0, :]

    # 叠加显示
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False)

    # 保存
    cv2.imwrite(str(SAVE_PATH), visualization)
    print(f">>> [成功] 热力图已保存为: {SAVE_PATH}")
    print("    请查看 heatmap.png，红色越深代表模型越确信那里是火。")

if __name__ == '__main__':
    main()