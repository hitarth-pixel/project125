from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
from PIL import Image

from pro1 import X_train_scaled 

X=np.load("image(2).npz")['arr_0']
y=pd.read_csv("labels.csv")['labels']
nclasses=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
classes=len(nclasses)

X_train,X_test,Y_train,Y_test=train_test_split(X,y,random_state=0,train_size=3500,test_size=500)
X_train_scaled=X_train/255
#Y_train_scaled=Y_train/255
clf=LogisticRegression(solver='saga',multi_class='multinomial').fit(X_train_scaled,Y_train)

def get_prediction(image):
    im_pil=Image.open(image)
    im_bw=im_pil.convert('L')
    im_bw_resized=im_bw.resize((22,22),Image.ANTIALIAS)
    pixel_filter=20
    min_pixel=np.percentile(im_bw_resized,pixel_filter)
    im_bw_resized_inverted=np.clip(im_bw_resized-min_pixel,0,255)
    max_pixel=np.max(im_bw_resized)
    im_bw_resized_inverted_scaled=np.asarray(im_bw_resized_inverted/max_pixel)
    testArray=np.array(im_bw_resized_inverted_scaled).reshape(1,660)
    testSample=clf.predict(testArray)
    print(testSample)
    return testSample[0]