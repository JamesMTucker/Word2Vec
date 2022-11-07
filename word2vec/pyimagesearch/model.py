import pyimagesearch.config as config

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def build_shallow_net():
    model = Sequential()
    # input dimensions = V (in our case it is 10)
    model.add(Dense(config.denseUnits, input_dim=10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model