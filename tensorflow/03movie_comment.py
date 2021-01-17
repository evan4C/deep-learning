import os

# INFO and WARNING messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

# 本教程演示了使用 Tensorflow Hub 和 Keras 进行迁移学习的基本应用。
# 下载 IMDB 数据集
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True
)

# 探索数据
"""每一个样本都是一个表示电影评论和相应标签的句子。该句子不以任何方式进行预处理。
# 标签是一个值为 0 或 1 的整数，其中 0 代表消极评论，1 代表积极评论。
"""
# 打印前十个样本
# train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
# print(train_examples_batch, '\n', train_labels_batch)

# 构建模型
"""
神经网络由堆叠的层来构建，这需要从三个主要方面来进行体系结构决策：
如何表示文本？
模型里有多少层？
每个层里有多少隐层单元（hidden units）？

本示例中，输入数据由句子组成。预测的标签为 0 或 1。
表示文本的一种方式是将句子转换为嵌入向量（embeddings vectors）。
我们可以使用一个预先训练好的文本嵌入（text embedding）作为首层，这将具有三个优点：
- 我们不必担心文本预处理
- 我们可以从迁移学习中受益
- 嵌入具有固定长度，更易于处理
 针对此示例我们将使用 TensorFlow Hub 中名为 google/tf2-preview/gnews-swivel-20dim/1 的一种预训练文本嵌入（text embedding）模型 。
"""
# 首先创建一个使用 Tensorflow Hub 模型嵌入（embed）语句的Keras层，并在几个输入样本中进行尝试。
# 请注意无论输入文本的长度如何，嵌入（embeddings）输出的形状都是：(num_examples, embedding_dimension)。
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)

# 现在让我们构建完整模型：
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

"""层按顺序堆叠以构建分类器：
第一层是 Tensorflow Hub 层。这一层使用一个预训练的保存好的模型来将句子映射为嵌入向量（embedding vector）。
该定长输出向量通过一个有 16 个隐层单元的全连接层（Dense）进行管道传输。
最后一层与单个输出结点紧密相连。使用 Sigmoid 激活函数，其函数值为介于 0 与 1 之间的浮点数，表示概率或置信水平。
"""

# 损失函数与优化器
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# 评估模型
# 我们来看下模型的表现如何。将返回两个值。损失值（loss）（一个表示误差的数字，值越低越好）与准确率（accuracy）。
results = model.evaluate(test_data.batch(512), verbose=2)