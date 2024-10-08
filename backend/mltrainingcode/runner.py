import os
os.environ["MRI_DATA_PATH"] = "/Users/kousthubhveturi/Desktop/brainClassify/backend/mltrainingcode/dataset"
import projlib as fc
import numpy as np
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context
import matplotlib.pyplot as plt
import keras
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.applications.resnet import ResNet152
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from keras.applications.efficientnet_v2 import EfficientNetV2L
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from keras.applications.nasnet import NASNetLarge
from sklearn.tree import DecisionTreeClassifier
from keras.applications.inception_v3 import InceptionV3
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

fc.train_classifier("VGG16","fc2",KNeighborsClassifier(n_neighbors=1))

fc.test_classifier("VGG16","fc2",KNeighborsClassifier(n_neighbors=1))
