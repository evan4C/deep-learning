import tensorflow as tf
from tensorflow import keras

new_model = tf.keras.models.load_model('saved_model/my_model')

new_model.summary()

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

# -1 意味着numpy将根据28*28以及原shape自动计算-1所在位置的维度
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# 还原的模型使用与原始模型相同的参数进行编译。 尝试使用加载的模型运行评估和预测：
# 评估还原的模型
loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

print(new_model.predict(test_images).shape)


# Keras使用 HDF5 标准提供了一种基本的保存格式。
# 创建并训练一个新的模型实例
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


model_HDF5 = create_model()
model_HDF5.fit(train_images, train_labels, epochs=5)

# 将整个模型保存为 HDF5 文件。
# '.h5' 扩展名指示应将模型保存到 HDF5。
model_HDF5.save('my_model.h5')

# 重新创建完全相同的模型，包括其权重和优化程序
new_model2 = tf.keras.models.load_model('my_model.h5')

# 显示网络结构
new_model2.summary()
loss, acc = new_model2.evaluate(test_images,  test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))
