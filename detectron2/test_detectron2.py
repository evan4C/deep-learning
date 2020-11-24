# 加载一些基础包以及设置logger
import detectron2
from detectron2.utils.logger import setup_logger

# 加载一些常用库
import numpy as np
import cv2
import random

# 使用detectron2已经训练好的神经网络模型
from detectron2 import model_zoo

# 通过指定配置文件来创建一个简单的预测器
from detectron2.engine import DefaultPredictor

# get_cfg函数用来得到detectron2的缺省配置
from detectron2.config import get_cfg

# Visualizer用来在原始图像上标注物体检测和分类结果
from detectron2.utils.visualizer import Visualizer

# metadata是指存放在数据集中的原始标注信息
from detectron2.data import MetadataCatalog


setup_logger()

# 导入准备好的图像
input_path = "test.jpg"
img = cv2.imread(input_path)

# 指定Faster-R-CNN模型的配置配置文件路径及神经网络参数文件的路径
model_file_path = model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
model_weights = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

# 创建一个detectron2配置
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'
# 要创建的模型的名称
cfg.merge_from_file(model_file_path)
# 为模型设置阈值
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# 加载模型需要的参数数据
cfg.MODEL.WEIGHTS = model_weights
# 基于配置创建一个预测器
predictor = DefaultPredictor(cfg)
# 利用这个预测器对准备好的图像进行分析并得到结果
outputs = predictor(img)

# 使用Visualizer对结果可视化，img[:, :, ::-1]实现RGB到BGR通道的相互转换，scale表示输出图像的缩放尺度
v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
# 将检测到的物体标注在图像上
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# 获得绘制的图像
result = v.get_image()[:, :, ::-1]
# 将影像保存到文件
cv2.imshow('result', result)
cv2.waitKey(0)
