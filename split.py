import os, random, shutil
from pathlib import Path
import cv2

# --- 1. 自动定位路径 ---
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR  = BASE_DIR / 'dataset_src'
DST_DIR  = BASE_DIR / 'fire_dataset'
RATIO    = 0.80
SEED     = 42
random.seed(SEED)

# --- 2. 自动清理旧数据 ---
if DST_DIR.exists():
    print(f'>>> [清理] 删除旧目录 {DST_DIR}...')
    shutil.rmtree(DST_DIR)

# --- 3. 收集图片 ---
print(f'>>> [搜索] 在 {SRC_DIR} 查找图片...')
if not SRC_DIR.exists():
    print(f"!!! 错误: 找不到源文件夹 {SRC_DIR}")
    exit()

suffix = ('.jpg', '.jpeg', '.png', '.bmp') 
imgs   = [p for p in SRC_DIR.rglob('*') if p.suffix.lower() in suffix]
random.shuffle(imgs)

if not imgs:
    print("!!! 错误: 未找到图片，请检查 dataset_src 文件夹。")
    exit()

n_train = int(len(imgs) * RATIO)
print(f'>>> [处理] 共找到 {len(imgs)} 张图片，开始切分...')

# --- 4. 核心处理循环 ---
for i, img in enumerate(imgs):
    split = 'train' if i < n_train else 'test'
    
    # 路径构造
    img_dst_dir = DST_DIR / 'images' / split
    lbl_dst_dir = DST_DIR / 'labels' / split
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)

    # 复制图片
    shutil.copy(img, img_dst_dir / img.name)

    # 处理标签
    txt_src = img.with_suffix('.txt')
    txt_dst = lbl_dst_dir / txt_src.name
    
    if txt_src.exists():
        shutil.copy(txt_src, txt_dst)
    else:
        # 【核心修正】伪标签：使用相对坐标 (0-1)
        # 0.5 0.5 1.0 1.0 代表：中心在正中，宽高占满全图
        with open(txt_dst, 'w') as f:
            f.write('0 0.5 0.5 1.0 1.0\n') 

print('>>> [完成] 数据集切分完毕！')
print(f'    训练集: {len(list((DST_DIR/"images/train").glob("*")))} 张')
print(f'    测试集: {len(list((DST_DIR/"images/test").glob("*")))} 张')