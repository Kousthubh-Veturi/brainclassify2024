import urllib
from urllib import request
import pathlib 
from pathlib import Path
import shutil
import os
import numpy as np
from numpy import save
import pathlib
import os
from pathlib import Path
import tensorflow as tf
import joblib
import keras
from keras import models 
from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications import resnet
from keras.models import Model
from keras.applications import efficientnet_v2
from keras.applications import nasnet
import numpy as np
import sklearn

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from keras.applications import inception_v3
import pickle as pkl
import glob
from typing import Any
import copy
from copy import deepcopy
import matplotlib.pyplot as plt
def generate_features(model_name:str,layer_name:str):
    '''
    before calling function, define following vars:
    environment variable to access directory for training data
    call environment variable: "MRI_DATA_PATH", points to the dataset
        
    '''
    
    if model_name == "VGG16":
        model = vgg16.VGG16(include_top=False)
        prep = vgg16.preprocess_input
        kept_layers = [] 
        for layer in model.layers:
            kept_layers.append(layer)
            if(layer.name==layer_name):
                break
        
        model = keras.Sequential(kept_layers) 
    else:
        model = vgg19.VGG19(include_top=False)
        prep = vgg19.preprocess_input
        kept_layers = []
        for layer in model.layers:
            kept_layers.append(layer)
            if(layer.name==layer_name):
                break
        
        model = keras.Sequential(kept_layers) 
    
  

    for split_name in ["Training","Testing"]:
        source = Path(os.environ["MRI_DATA_PATH"],split_name)
        class_dirs = sorted(source.glob("[!.]*"))
        for class_dir in class_dirs:
            for file_path in class_dir.glob("[!.]*"): 
                #dest = Path("artifacts/vgg_features/",split_name,class_dir.name,file_path.stem).with_suffix(".npy") 
                dest = Path("artifacts/",model_name,layer_name,"features",split_name,class_dir.name,file_path.stem).with_suffix(".npy")  
                image = tf.keras.preprocessing.image.load_img(file_path,target_size=(224,224))
                input_arr = prep(tf.keras.preprocessing.image.img_to_array(image))[np.newaxis,...]
                features = model.predict(input_arr,verbose=0)[0] 
                if features.ndim == 3:
                    height = features.shape[0]
                    width = features.shape[1]
                    features = features[height//2,width//2] 
                    #features = features.flatten()
                dest.parent.mkdir(parents=True,exist_ok=True)
                np.save(dest,features)
 

def train_classifier(model_name:str,layer_name:str,classifier:Any): 
    features = []
    labels = []
    source = Path("artifacts/"+model_name+"/"+layer_name+"/features/training")
    
    class_dirs = sorted(source.glob("[!.]*"))
   
    for i,class_dir in enumerate(sorted(source.glob("[!.]*"))):

        for file_path in class_dir.glob("[!.]*"):
            features.append(np.load(file_path))
            labels.append(i)
                
 
    
    features = np.array(features)
    labels = np.array(labels)



    #knc = KNeighborsClassifier(n_neighbors=1)
    classifier = deepcopy(classifier)
    classifier.fit(features,labels) 
    
    dest = Path("artifacts/classifiers/",model_name,layer_name,str(classifier)).with_suffix(".joblib") 
    dest.parent.mkdir(parents=True,exist_ok=True)
    #dest.write_bytes(pkl.dumps(classifier)) 
    joblib.dump(classifier,dest)
    print("saved model") 

def test_classifier(model_name:str,layer_name:str,classifier:Any):
    features = []
    labels = []
    source = Path("artifacts/"+model_name+"/"+layer_name+"/features/testing")

    class_dirs = sorted(source.glob("[!.]*"))
    for i,class_dir in enumerate(sorted(source.glob("[!.]*"))):

        for file_path in class_dir.glob("[!.]*"):
            features.append(np.load(file_path))
            labels.append(i)
     
    labels = np.array(labels)
    features = np.array(features)
    location_of_classifier = Path("artifacts/classifiers/",model_name,layer_name,str(classifier)).with_suffix(".joblib") 
    #model = pkl.load(open(location_of_classifier, 'rb')) 
    #model = pkl.loads(location_of_classifier.read_bytes())
    model = joblib.load(location_of_classifier)
    predictions = model.predict(features) 


    print(np.mean(labels==predictions)*100 , "%")
    pred_counts = np.zeros((4,4))
    for label,prediction in zip(labels,predictions):
        pred_counts[label,prediction]+=1

    print(pred_counts)
    pred_rates = (pred_counts/(np.sum(pred_counts,axis=1,keepdims=True)))
    print(pred_rates)

    figdest = Path("artifacts/figures/",model_name,layer_name,str(classifier),"confusion_matrix.pdf")
    figdest.parent.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(3,3))
    plt.matshow(pred_rates,cmap="RdYlGn")
    plt.savefig(figdest)
    plt.close()