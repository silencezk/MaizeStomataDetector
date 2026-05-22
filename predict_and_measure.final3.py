#!/usr/bin/env python3
"""
气孔检测与测量脚本
用法：
    python predict_and_measure.final3.py --image_dir ./example_data/images --output results.csv --scale 0.087
"""

import os
import cv2
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='气孔检测与测量')
    
    # 必选参数
    parser.add_argument('--model', type=str, 
                        default='runs/segment/maize_stomatal_seg3/weights/best.pt',
                        help='模型文件路径 (默认: runs/segment/maize_stomatal_seg3/weights/best.pt)')
    
    parser.add_argument('--image_dir', type=str, 
                        default='./example_data/images',
                        help='输入图片文件夹路径 (默认: ./example_data/images)')
    
    parser.add_argument('--output_csv', type=str, 
                        default='./results/stomata_measurements.csv',
                        help='输出CSV文件路径 (默认: ./results/stomata_measurements.csv)')
    
    # 可选参数
    parser.add_argument('--scale', type=float, 
                        default=0.087,
                        help='像素转微米的比例因子 (默认: 0.087 um/pixel)')
    
    parser.add_argument('--conf', type=float, 
                        default=0.25,
                        help='检测置信度阈值 (默认: 0.25)')
    
    parser.add_argument('--imgsz', type=int, 
                        default=640,
                        help='模型输入图像尺寸 (默认: 640)')
    
    parser.add_argument('--save_per_image', action='store_true',
                        help='是否保存每张图片的独立CSV文件')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"❌ 错误: 模型文件不存在: {args.model}")
        print("💡 提示: 请使用 --model 参数指定正确的模型路径")
        return
    
    # 检查图片目录是否存在
    if not os.path.exists(args.image_dir):
        print(f"❌ 错误: 图片目录不存在: {args.image_dir}")
        print("💡 提示: 请使用 --image_dir 参数指定包含图片的文件夹路径")
        return
    
    # 创建输出目录
    output_dir = Path(args.output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("🔬 气孔检测与测量工具")
    print("="*50)
    print(f"📦 模型: {args.model}")
    print(f"📁 图片目录: {args.image_dir}")
    print(f"📄 输出文件: {args.output_csv}")
    print(f"📏 比例因子: {args.scale} um/pixel")
    print(f"🎯 置信度阈值: {args.conf}")
    print("="*50)
    
    # 加载模型
    print("🔄 加载模型中...")
    model = YOLO(args.model)
    
    # 获取所有图片
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(Path(args.image_dir).glob(ext)))
    image_paths = sorted(image_paths)
    
    if len(image_paths) == 0:
        print(f"❌ 错误: 在 {args.image_dir} 中没有找到图片文件")
        print("💡 支持的格式: jpg, jpeg, png, bmp, tiff")
        return
    
    print(f"📷 找到 {len(image_paths)} 张图片")
    
    all_results = []
    
    for img_idx, img_path in enumerate(image_paths, 1):
        print(f"\n🔄 [{img_idx}/{len(image_paths)}] 处理: {img_path.name}")
        
        # 预测
        results = model(img_path, imgsz=args.imgsz, conf=args.conf, verbose=False, retina_masks=True)
        result = results[0]
        
        # 读取原图
        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            print(f"  ⚠️ 无法读取图片: {img_path}")
            continue
        h_orig, w_orig = orig_img.shape[:2]
        
        measurements = []
        
        if result.masks is not None:
            for mask_tensor in result.masks.data:
                mask_np = mask_tensor.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                cnt = max(contours, key=cv2.contourArea)
                
                # 面积计算
                area_pixels = cv2.contourArea(cnt)
                area_um2 = area_pixels * (args.scale ** 2)
                
                # 长宽计算
                if len(cnt) < 5:
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = box.astype(np.int32)
                    d1 = np.linalg.norm(box[0] - box[1])
                    d2 = np.linalg.norm(box[1] - box[2])
                    length_px = max(d1, d2)
                    width_px = min(d1, d2)
                else:
                    try:
                        ellipse = cv2.fitEllipse(cnt)
                        major_axis = max(ellipse[1])
                        minor_axis = min(ellipse[1])
                        length_px = major_axis
                        width_px = minor_axis
                    except cv2.error:
                        rect = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(rect)
                        d1 = np.linalg.norm(box[0] - box[1])
                        d2 = np.linalg.norm(box[1] - box[2])
                        length_px = max(d1, d2)
                        width_px = min(d1, d2)
                
                length_um = length_px * args.scale
                width_um = width_px * args.scale
                measurements.append((length_um, width_um, area_um2))
        
        stomata_count = len(measurements)
        print(f"  ✅ 检测到 {stomata_count} 个气孔")
        
        # 创建 DataFrame
        if measurements:
            df_img = pd.DataFrame(measurements, columns=["length_um", "width_um", "area_um2"])
            df_img.insert(0, "image_name", img_path.name)
            df_img.insert(1, "stomata_id", range(len(df_img)))
            df_img["stomata_count_in_image"] = stomata_count
            all_results.append(df_img)
            
            # 可选：每张图片单独保存
            if args.save_per_image:
                per_img_csv = output_dir / f"{img_path.stem}_results.csv"
                df_img.to_csv(per_img_csv, index=False)
                print(f"  💾 已保存: {per_img_csv}")
    
    # 保存汇总结果
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df = final_df[[
            "image_name", "stomata_id", "stomata_count_in_image",
            "length_um", "width_um", "area_um2"
        ]]
        final_df.to_csv(args.output_csv, index=False)
        print(f"\n🎉 完成！结果保存至: {args.output_csv}")
        print(f"📊 共处理 {len(final_df)} 个气孔")
    else:
        print("\n⚠️ 未检测到任何气孔。")

if __name__ == '__main__':
    main()