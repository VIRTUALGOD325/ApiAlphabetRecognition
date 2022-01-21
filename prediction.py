import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps


X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']

print(pd.Series(y).value_counts())

classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

nclasses = len(classes)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=9,train_size=3500,test_size=500)


'''
X,y = fetch_openml("mnist_784",version =1,return_X_y=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=10,train_size=7500,test_size=2500)
'''

X_train_scaled = X_train/255
X_test_scaled = X_test/255


lr = LogisticRegression(solver="saga",multi_class="multinomial").fit(X_train_scaled,y_train)

def get_prediction(image):
    img_pil = Image.open(image)
    img_bw = img_pil.convert("L")
    img_bw_resize = img_bw.resize((28,28),image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(img_bw_resize,pixel_filter)
    img_resize_inverted_scaled = np.clip(img_bw_resize - min_pixel,0,255)
    max_pixel = np.max(img_bw_resize)
    img_resize_inverted_scaled = np.asarray(img_resize_inverted_scaled)/max_pixel
    test_sample = np.array(img_resize_inverted_scaled).reshape(1,784)
    test_predict = lr.predict(test_sample)
    return test_predict[0] 



