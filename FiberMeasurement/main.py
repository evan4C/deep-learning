from data_processor import load_and_preprocess_image
from visualizer import visualizer, analysis
from sem_model import sem_model
import tensorflow as tf
import glob
import numpy as np
import os


AUTOTUNE = tf.data.experimental.AUTOTUNE
os.chdir('/Users/lifan/Documents/GitHub/DeepLearning/FiberMeasurement/data')

train_dir = './test_images'

image_height = 960
image_width = 1280
img_size = 192
num_img = 60

train_label = np.genfromtxt(train_dir + '/label.csv', delimiter=',')
Rx = img_size / image_width
Ry = img_size / image_height
for i in train_label:
    i[0::3] *= Rx
    i[1::3] *= Ry
print('train_label:', train_label.shape)

train_img_paths = sorted(glob.glob(train_dir + '/*.jpg'))
path_ds = tf.data.Dataset.from_tensor_slices(train_img_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

#for n, image in enumerate(image_ds.take(2)):
#    visualizer(image, train_label[n])
train_data = np.zeros((num_img, img_size, img_size, 1))
for i, img in enumerate(image_ds):
    train_data[i] = img

print('train_data:', train_data.shape)

model, history = sem_model(train_data, train_label)
history_dict = history.history

analysis()

# pred = model.predict(train_data[0])
