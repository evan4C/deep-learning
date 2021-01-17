import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# 每个图像都会被映射到一个标签。由于数据集不包括类名称，请将它们存储在下方，供稍后绘制图像时使用：
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# check the first image
def check_img():
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()


# check_img()
# 将这些值缩小至 0 到 1 之间，然后将其馈送到神经网络模型。为此，请将这些值除以 255。
train_images = train_images / 255.0
test_images = test_images / 255.0


# 为了验证数据的格式是否正确，显示训练集中的前 25 个图像，并在每个图像下方显示类名称。
def confirm():
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()


# confirm()

# 设置层
# 构建神经网络需要先配置模型的层，然后再编译模型。
# Flatten 将图像格式从二维数组（28 x 28 像素）转换成一维数组（28 x 28 = 784 像素）
# 展平像素后，网络会包括两个 tf.keras.layers.Dense 层的序列。它们是密集连接或全连接神经层。
# 第一个 Dense 层有 128 个节点（或神经元）
# 第二个（也是最后一个）层会返回一个长度为 10 的 logits 数组。
# 每个节点都包含一个得分，用来表示当前图像属于 10 个类中的哪一类。
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# 编译模型
# 在准备对模型进行训练之前，还需要再对其进行一些设置。
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练模型
# 1. 将训练数据馈送给模型。
# 2. 模型学习将图像和标签关联起来
# 3. 要求模型对测试集进行预测
# 4. 验证预测是否与test_labels数组中的标签相匹配

# 调用 model.fit 方法开始训练
model.fit(train_images, train_labels, epochs=10)

# 评估准确率
# 比较模型在测试数据集上的表现
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# 进行预测
# 在模型经过训练后，您可以使用它对一些图像进行预测。模型具有线性输出，即 logits。
# 您可以附加一个 softmax 层，将 logits 转换成更容易理解的概率。
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

# predictions中包括对test集中每个图像的预测，
# 例如predictions[0]是一个包含 10 个数字的数组。它们代表模型对 10 种不同服装中每种服装的“置信度”。
predictions = probability_model.predict(test_images)


# 绘制图形以及其预测结果，结果显示在图形下方，蓝色为预测成功，红色为预测失败，括号内为真实结果
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)  # 得到置信度最高的结果的索引，再通过class_name转换成名称
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


# 绘制结果柱状图，横坐标为10个类别，纵坐标为每个类别的置信度，蓝色为预测成功，红色为预测失败
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# 验证预测结果
def check(i):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()

"""
# 绘制图像的预测结果
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
"""

# 使用训练好的模型
img = test_images[1]
# tf.keras 模型经过了优化，可同时对一个批或一组样本进行预测。因此，即便只使用一个图像，也需要将其添加到列表中：
img = (np.expand_dims(img, 0))
predictions_single = probability_model.predict(img)
plot_value_array(1, predictions_single[0], test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()
