from keras import layers
from keras import models


def sem_model(x_train, y_train):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding='same', use_bias=False, input_shape=(192, 192, 1)))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(96, (3, 3), padding='same', use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), padding='same', use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(512, (3, 3), padding='same', use_bias=False))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(15))
    model.summary()

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])

    history = model.fit(x_train, y_train, epochs=50, batch_size=1, validation_split=0.2)
    model.save('model.h5')

    return model, history
