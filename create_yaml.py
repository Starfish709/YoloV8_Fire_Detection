from pathlib import Path
import sys

# --- 自动定位路径 ---
BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / 'fire_dataset'
YAML_NAME    = BASE_DIR / 'fire.yaml'

print(f">>> [检查] 数据集路径: {DATASET_ROOT}")

if not DATASET_ROOT.exists():
    print(f"!!! 错误: 找不到数据集，请先运行 split.py")
    sys.exit(1)

# --- 写入配置 ---
# 注意：nc: 1 代表只有 'fire' 一类
yaml_content = f"""
path: {DATASET_ROOT.resolve()}
train: images/train
val: images/test
test: images/test

nc: 1
names: ['fire']
"""

with open(YAML_NAME, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print(f'>>> [成功] 已生成配置文件: {YAML_NAME}')
print(yaml_content)