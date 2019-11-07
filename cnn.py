import numpy as np
import generate_samples as gen

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, Flatten, Dropout, MaxPooling2D, LeakyReLU
from keras.utils import to_categorical

from sklearn.metrics import confusion_matrix

def prepare_data(n):
    data = gen.generate_samples(n, True, n_contributors=3)
    X = np.reshape(data[0], (n, 2, 10, 1))
    y = data[1]
    y = to_categorical(y)
    return X, y

def make_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=2, padding='same', input_shape=(2, 10, 1)))
    model.add(LeakyReLU())
    model.add(Conv2D(8, kernel_size=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU())
    model.add(Dense(64))
    model.add(LeakyReLU())
    model.add(Dense(2, activation='softmax'))
    return model

X_train, y_train = prepare_data(10000)
X_test, y_test = prepare_data(1000)

model = make_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=256, epochs=100)

print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test, verbose=0), axis=1)))
print(model.test_on_batch(X_test, y_test))