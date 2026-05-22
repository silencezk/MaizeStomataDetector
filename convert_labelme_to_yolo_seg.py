import os
import json
from PIL import Image, ImageDraw

def create_mask_from_json(json_path, output_mask_dir):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取 {json_path}: {e}")
        return
    
    img_h, img_w = data.get('imageHeight'), data.get('imageWidth')
    if not img_h or not img_w:
        print(f"⚠️ 跳过 {json_path}：缺少 imageHeight 或 imageWidth")
        return

    mask = Image.new('L', (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    stomatal_count = 0

    for shape in data.get('shapes', []):
        if shape.get('label') == 'stomatal' and shape.get('shape_type') == 'polygon':
            points = [(int(x), int(y)) for x, y in shape['points']]
            if len(points) >= 3:
                draw.polygon(points, fill=255)
                stomatal_count += 1

    base_name = os.path.splitext(os.path.basename(json_path))[0]
    mask_path = os.path.join(output_mask_dir, base_name + '.png')
    mask.save(mask_path)
    print(f"✅ {mask_path} 已保存 ({stomatal_count} 个气孔)")

def main():
    splits = ['train', 'val']
    for split in splits:
        json_dir = f'labels/{split}'
        mask_dir = f'segmentation_masks/{split}'
        os.makedirs(mask_dir, exist_ok=True)
        
        if not os.path.exists(json_dir):
            print(f"❌ 目录不存在: {json_dir}")
            continue
            
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        if not json_files:
            print(f"⚠️ {json_dir} 中没有 .json 文件！")
            continue
            
        print(f"\n📁 处理 {split} 集 ({len(json_files)} 个文件):")
        for json_file in json_files:
            json_path = os.path.join(json_dir, json_file)
            create_mask_from_json(json_path, mask_dir)

if __name__ == '__main__':
    main()
EOF
