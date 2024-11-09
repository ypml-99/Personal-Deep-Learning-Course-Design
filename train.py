from ultralytics import YOLO
import torch

torch.cuda.empty_cache()

# 加载模型
model = YOLO("yolov10n.yaml")

model.train(
    data="voc_data.yaml",
    epochs=100,                 # 增加训练轮数
    imgsz=1024,                  # 图像尺寸
    batch=32,                   # 增大batch size
    lr0=0.002,                  # 提高初始学习率
    lrf=0.05,                   # 调整学习率衰减为0.05
    weight_decay=0.0005,        # 添加权重衰减，防止过拟合
    momentum=0.937,             # 优化动量，提高收敛速度
    mosaic=0.5,                 # 增强Mosaic数据增强强度
    mixup=0.1,                  # 增强Mixup数据增强强度
    multi_scale=True,           # 启用多尺度训练
    pretrained=True,            # 使用预训练模型
    amp=True,                   # 启用混合精度训练
    box=0.05,                   # 调整box损失权重
    cls=0.4,                    # 提高分类损失权重，增加正则化
    dfl=0.6,                    # 调整DIOU损失权重
    freeze=10,                  # 冻结前10层，提高模型适应性
    translate=0.1,              # 启用平移增强
    scale=0.5,                  # 调整尺度变化
    shear=0.2,                  # 增加剪切强度
    auto_augment="randaugment"  # 自动增强
)
