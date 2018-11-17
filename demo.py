#coding:utf-8
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from google.colab import files
from keras.models import Model
from keras import backend as K
uploaded = files.upload()
import keras
from keras.applications.vgg16 import VGG16

model = InceptionV3(include_top=True, weights="imagenet", input_tensor=None, input_shape=None)
model.summary()

IMG_PATH = "horse.jpg" #画像ファイルパスを各自入力

img = image.load_img(IMG_PATH, target_size=(224,224,3))
print(img)
x = image.img_to_array(img)
X = np.array([preprocess_input(x)])
print("X.shape {}".format(X.shape))



intermediante_layer_model = Model(inputs=model.input, outputs=model.get_layer("max_pooling2d_41").output)
y = intermediante_layer_model.predict(X)
print(y.shape)



get_layer_output = K.function([model.layers[0].input],[model.layers[21].output])
y = get_layer_output([X,0])[0]
print(y.shape)

#coding:utf-8
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from google.colab import files
from keras.models import Model
from keras import backend as K
uploaded = files.upload()
import keras
from keras.applications.vgg16 import VGG16

model = InceptionV3(include_top=True, weights="imagenet", input_tensor=None, input_shape=None)
model.summary()

IMG_PATH = "horse.jpg" #画像ファイルパスを各自入力

img = image.load_img(IMG_PATH, target_size=(224,224,3))
print(img)
x = image.img_to_array(img)
X = np.array([preprocess_input(x)])
print("X.shape {}".format(X.shape))



intermediante_layer_model = Model(inputs=model.input, outputs=model.get_layer("max_pooling2d_41").output)
y = intermediante_layer_model.predict(X)
print(y.shape)



get_layer_output = K.function([model.layers[0].input],[model.layers[21].output])
y = get_layer_output([X,0])[0]
print(y.shape)