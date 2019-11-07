import numpy as np
import generate_samples as gen

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LeakyReLU
from keras.utils import to_categorical

def prepare_data(n):
    data = gen.generate_samples(n, True)
    X = np.reshape(data[0], (n, 2, 10, 1))
    y = data[1]
    y = to_categorical(y)
    return X, y

def make_model():
    global model
    model = Sequential()
    model.add(Conv2D(16, kernel_size=2, padding='same', input_shape=(2, 10, 1)))
    model.add(LeakyReLU())
    model.add(Conv2D(8, kernel_size=2, padding='same'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(Dense(16))
    model.add(LeakyReLU())
    model.add(Dense(8))
    model.add(LeakyReLU())
    model.add(Dense(2, activation='softmax'))
    return model

X_train, y_train = prepare_data(10000)
X_test, y_test = prepare_data(1000)

model = make_model()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=512, epochs=20)
print(model.test_on_batch(X_test, y_test))

