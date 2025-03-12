import keras
from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense




entrada = Input((153, 153, 3), name = 'image')
x=entrada

for i in range(0, 5):
  nb_filters = 2**(4 + i)
  x = Conv2D(nb_filters, 3, activation='relu')(x)
  x = BatchNormalization()(x)
  x = MaxPool2D(2)(x)

x = Flatten()(x)

x = Dense(256, activation= 'relu')(x)

class_out = Dense(9, activation='softmax', name='class_out')(x)
box_out = Dense(2, name='box_out')(x)


model = keras.models.Model(entrada, [class_out, box_out])
model.summary()

from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
