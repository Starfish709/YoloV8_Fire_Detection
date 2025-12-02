from ultralytics import YOLO
from pathlib import Path
import torch


# 自动定位当前脚本所在文件夹
BASE_DIR    = Path(__file__).resolve().parent
YAML_PATH   = BASE_DIR / 'fire.yaml'

# M3 支持 MPS 加速 (Metal Performance Shaders)
# 如果 MPS 可用就用 MPS，否则回退到 CPU
DEVICE      = 'mps' if torch.backends.mps.is_available() else 'cpu'

# M3 16GB 内存足够大，可以将 Batch 设为 32 甚至 64
BATCH       = 64 
IMGSZ       = 640
EPOCHS      = 20
NAME        = 'yolov8n_fire_m3' 

if __name__ == '__main__':
    if not YAML_PATH.exists():
        print("->>create_yaml.py不存在")
        exit()

    print(f">>> 运行设备: {DEVICE} ，mps加速已启用")
    
    # 加载模型
    model = YOLO('yolov8n.pt')

    # 开始训练
    model.train(
        data        = str(YAML_PATH),
        epochs      = EPOCHS,
        imgsz       = IMGSZ,
        batch       = BATCH,
        device      = DEVICE,
        project     = str(BASE_DIR / 'runs/fire'),
        name        = NAME,
        exist_ok    = True,
        pretrained  = True,
        optimizer   = 'auto',   # 让 YOLO 自动选择优化器
        patience    = 30,       # 30轮不提升就早停，防止过拟合
        workers     = 4,        # 基于m3芯片，使用4个worker 加载数据
        val         = True,
        plots       = True,
        amp         = True      # 混合精度训练，M3 支持，能加速
    )

    print('\n>>> 训练完成！')
    print(f'>>> 最佳权重: runs/fire/{NAME}/weights/best.pt')