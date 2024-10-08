import sys

print(sys.path)
import keras
import joblib
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import load_model, Model
import numpy as np


model = ('artifacts/classifiers/VGG16/fc2/KNeighborsClassifier(n_neighbors=1).joblib')
vggmodel = VGG16(weights="imagenet", include_top=True)
print(vggmodel)

model_layer_output = vgg16_model.get_layer('fc2').output
featureextract = Model(inputs=vgg16_model.input, outputs=model_layer_output)

img_path = sys.argv[1]

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

features = featureextract.predict(img_path)

np.save('artifacts/features.npy', features)

final_prediction = model.predict(features)
prediction_class = np.argmax(final_prediction[0])
classes = ["glioma","meningioma","notumor","pituitary"]
print(classes[prediction_class])
 

