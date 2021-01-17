"""模型可以在训练期间和训练完成后进行保存。这意味着模型可以从任意中断中恢复，并避免耗费比较长的时间在训练上。
保存也意味着您可以共享您的模型，而其他人可以通过您的模型来重新创建工作。"""

import os

import tensorflow as tf
from tensorflow import keras

# 获取示例数据集
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

# -1 意味着numpy将根据28*28以及原shape自动计算-1所在位置的维度
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# 定义模型
# 定义一个简单的序列模型
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


# 创建一个基本的模型实例
model1 = create_model()

# 显示模型的结构
model1.summary()

# 在训练期间保存模型（以 checkpoints 形式保存）
# 您可以使用训练好的模型而无需从头开始重新训练，或在您打断的地方开始训练，以防止训练过程没有保存。
# tf.keras.callbacks.ModelCheckpoint 允许在训练的过程中和结束时回调保存的模型。

# Checkpoint 回调用法
checkpoint_path = "training_1/cp.ckpt"

# 创建一个保存模型权重的回调
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# 使用新的回调训练模型
model1.fit(train_images,
           train_labels,
           epochs=10,
           validation_data=(test_images, test_labels),
           callbacks=[cp_callback])  # 通过回调训练

# 创建一个新的未经训练的模型。仅恢复模型的权重时，必须具有与原始模型具有相同网络结构的模型。
# 由于模型具有相同的结构，您可以共享权重，尽管它是模型的不同实例。
# 创建一个基本模型实例
model2 = create_model()
model2.summary()

# 评估模型
loss, acc = model1.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# 然后从 checkpoint 加载权重并重新评估：
model2.load_weights(checkpoint_path)

# 重新评估模型
loss, acc = model2.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# checkpoint 回调选项
# 回调提供了几个选项，为 checkpoint 提供唯一名称并调整 checkpoint 频率。
# 训练一个新模型，每五个 epochs 保存一次唯一命名的 checkpoint ：

# 在文件名中包含 epoch (使用 `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"

# 创建一个回调，每 5 个 epochs 保存模型的权重
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5)

# 创建一个新的模型实例
model3 = create_model()

# 使用 `checkpoint_path` 格式保存权重
model3.save_weights(checkpoint_path.format(epoch=0))
checkpoint_dir = os.path.dirname(checkpoint_path)

# 使用新的回调训练模型
model3.fit(train_images,
           train_labels,
           epochs=50,
           callbacks=[cp_callback],
           validation_data=(test_images, test_labels),
           verbose=0)

latest = tf.train.latest_checkpoint(checkpoint_dir)

# 创建一个新的模型实例
model4 = create_model()

# 加载以前保存的权重
model4.load_weights(latest)

# 重新评估模型
loss, acc = model4.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# 上述代码将权重存储到 checkpoint—— 格式化文件的集合中，这些文件仅包含二进制格式的训练权重。 Checkpoints 包含：
# 一个或多个包含模型权重的分片。
# 索引文件，指示哪些权重存储在哪个分片中。
# 如果你只在一台机器上训练一个模型，你将有一个带有后缀的碎片： .data-00000-of-00001

# 手动保存权重
# 您将了解如何将权重加载到模型中。使用 Model.save_weights 方法手动保存它们同样简单。
# 默认情况下， tf.keras 和 save_weights 特别使用 TensorFlow checkpoints 格式 .ckpt 扩展名
# 和 ( 保存在 HDF5 扩展名为 .h5 保存并序列化模型 )：
# 保存权重
model4.save_weights('./checkpoints/my_checkpoint')

# 创建模型实例
model5 = create_model()

# 恢复权重
model5.load_weights('./checkpoints/my_checkpoint')

# 评估模型
loss, acc = model5.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# 保存整个模型
# 调用 model.save 将保存模型的结构，权重和训练配置保存在单个文件/文件夹中。
# 这可以让您导出模型，以便在不访问原始 Python 代码的情况下使用它。因为优化器状态（optimizer-state）已经恢复，您可以从中断的位置恢复训练。
# 整个模型可以以两种不同的文件格式（SavedModel 和 HDF5）进行保存。需要注意的是 TensorFlow 的 SavedModel 格式是 TF2.x. 中的默认文件格式。但
# 是，模型仍可以以 HDF5 格式保存。下面介绍了以两种文件格式保存整个模型的更多详细信息。
# 保存完整模型会非常有用——您可以在 TensorFlow.js（Saved Model, HDF5）加载它们，然后在 web 浏览器中训练和运行它们，或者使用 TensorFlow Lite 将它们转换为在移动设备上运行。
# SavedModel 格式
# SavedModel 格式是序列化模型的另一种方法。以这种格式保存的模型，可以使用 tf.keras.models.load_model 还原，并且模型与 TensorFlow Serving 兼容。
# 创建并训练一个新的模型实例。
model6 = create_model()
model6.fit(train_images, train_labels, epochs=5)

# 将整个模型另存为 SavedModel。
model6.save('saved_model/my_model')

