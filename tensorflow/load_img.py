# 本教程提供一个如何使用 tf.data 加载图片的简单例子。
import pathlib
import random
import os
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

# 下载并检查数据集
# 检索图片
# 在你开始任何训练之前，你将需要一组图片来教会网络你想要训练的新类别。
# 你已经创建了一个文件夹，存储了最初使用的拥有创作共用许可的花卉照片。

data_root_orig = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    fname='flower_photos', untar=True)
data_root = pathlib.Path(data_root_orig)
print(data_root)
for item in data_root.iterdir():
    print(item)

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)

# 检查图片
attributions = (data_root/"LICENSE.txt").open(encoding='utf-8').readlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)

