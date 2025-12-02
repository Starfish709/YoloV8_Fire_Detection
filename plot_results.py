import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# --- 1. 自动定位路径 (防止找不到文件) ---
BASE_DIR = Path(__file__).resolve().parent
# 路径对应最新的训练结果
RESULTS_CSV = BASE_DIR / 'runs/fire/yolov8n_fire_m3/results.csv' 
SAVE_PATH   = BASE_DIR / 'training_report.png'

def plot_curves():
    # 检查文件是否存在
    if not RESULTS_CSV.exists():
        print(f"!!! 错误: 找不到结果文件: {RESULTS_CSV}")
        print("请确认你已经完成了训练，或者修改代码中的 RESULTS_CSV 路径。")
        return

    # 读取数据
    try:
        df = pd.read_csv(RESULTS_CSV)
        # 【关键优化】去除列名两端的空格，YOLO生成的csv经常带有空格导致KeyError
        df.columns = df.columns.str.strip()
        print(f">>> 成功读取数据，共 {len(df)} 轮。")
        print(f">>> 数据列名: {df.columns.tolist()}")
    except Exception as e:
        print(f"!!! 读取 CSV 失败: {e}")
        return

    # 设置画图风格
    plt.style.use('ggplot') # 使用更好看的样式
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # 获取 Epoch (X轴)
    epochs = df['epoch']

    # 图1：损失曲线 (完整记录训练和验证损失 ) 
    
    if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
        ax[0].plot(epochs, df['train/box_loss'], label='Train Box Loss', linewidth=2)
        ax[0].plot(epochs, df['val/box_loss'], label='Val Box Loss', linewidth=2, linestyle='--')
        ax[0].set_title('Loss Curve (Box)', fontsize=14)
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[0].grid(True)
    else:
        ax[0].set_title('Loss Columns Not Found')
        print("警告: 未在CSV中找到 box_loss 相关列")

    # --- 图2：准确率曲线 (对应考核：准确率曲线 +5分 & 验证准确率达标 +5分) ---
    # YOLO的目标检测准确率主要看 mAP50 和 mAP50-95
    # 考核要求的 96% 通常指的是 metrics/mAP50(B)
    
    metric_col = 'metrics/mAP50(B)'
    if metric_col not in df.columns:
        # 尝试找不带(B)的旧版本写法
        metric_col = 'metrics/mAP50'
    
    if metric_col in df.columns:
        final_acc = df[metric_col].iloc[-1] # 最后一轮的准确率
        
        ax[1].plot(epochs, df[metric_col], label='mAP@0.5 (Accuracy)', color='green', linewidth=2)
        
        # 额外画出 mAP50-95 作为参考（体现专业性）
        map95_col = 'metrics/mAP50-95(B)'
        if map95_col in df.columns:
            ax[1].plot(epochs, df[map95_col], label='mAP@0.5:0.95', color='orange', alpha=0.7)

        ax[1].set_title(f'Accuracy Curve (Final mAP@0.5: {final_acc:.2%})', fontsize=14)
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Score (0-1)')
        
        # 画一条 0.96 的红线，方便老师一眼看出是否达标
        ax[1].axhline(y=0.96, color='red', linestyle=':', label='Target (96%)')
        
        ax[1].legend()
        ax[1].grid(True)
        
        if final_acc >= 0.96:
            print(f">>> 恭喜！最终准确率 {final_acc:.2%} 已达到 96% 以上考核标准。")
        else:
            print(f">>> 注意：最终准确率 {final_acc:.2%} 尚未达到 96%，请检查模型或增加训练轮数。")
            
    else:
        ax[1].set_title('Metrics Column Not Found')
        print(f"警告: 未找到 {metric_col}")

    # 保存图片
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300) # 300 dpi高清保存
    print(f">>> 图表已保存为: {SAVE_PATH}")
    # plt.show() # 如果是在服务器无界面环境，注释掉这行

if __name__ == '__main__':
    plot_curves()