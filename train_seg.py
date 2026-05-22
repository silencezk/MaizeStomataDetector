#!/usr/bin/env python3
"""
YOLOv8 气孔分割训练脚本
使用方法: python train_seg.py
"""

import yaml
import torch
from ultralytics import YOLO

# 打印配置文件内容
print("\n🔍 加载配置文件 data.yaml:")
with open('data.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
    for k, v in cfg.items():
        print(f"  {k}: {v}")

# 自动检测设备
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"\n🖥️ 使用设备: {device}")

# 加载预训练模型（会自动下载 yolov8n-seg.pt）
print("📦 加载预训练模型...")
model = YOLO('yolov8n-seg.pt')

# 开始训练
print("🚀 开始训练...")
model.train(
    data='data.yaml',      # 配置文件路径（相对路径）
    epochs=100,            # 训练轮数
    imgsz=640,             # 输入图像尺寸
    batch=8,               # 批次大小（显存小可改为 4 或 2）
    name='maize_stomatal_seg',  # 输出文件夹名称
    device=device,         # 自动选择 GPU 或 CPU
    verbose=True           # 显示训练详情
)

print("\n✅ 训练完成！")
print("📁 模型保存在: runs/segment/maize_stomatal_seg/weights/best.pt")
