# Maize Stomata Detector

基于 YOLOv8-seg 的玉米气孔检测与表型测量工具。

## 功能

- 将 LabelMe 标注的 JSON 文件转换为 YOLO 分割训练格式
- 训练气孔分割模型
- 批量预测并测量气孔的长度、宽度、面积

## 环境要求

- Python 3.8+
- ultralytics, opencv-python, pandas, pillow

## 快速开始

1. 克隆仓库
2. 安装依赖：`pip install ultralytics opencv-python pandas pillow`
3. 准备数据：将标注好的 JSON 文件放入 `labels/train` 和 `labels/val`
4. 生成掩码：`python convert_labelme_to_yolo_seg.py`
5. 训练模型：`python train_seg.py`
6. 预测测量：`python predict_and_measure.final3.py --image_dir ./example_data/images --output results.csv`

## 参数说明

- `predict_and_measure.final3.py` 支持命令行参数：
  - `--model`：模型路径
  - `--image_dir`：图片目录
  - `--output_csv`：输出 CSV 路径
  - `--scale`：像素/微米比例因子
  - `--conf`：置信度阈值

## 示例数据

`example_data/` 目录提供了一张示例图片和对应的 JSON 标注，可以用于测试。

## 许可证

MIT
