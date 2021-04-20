import cv2
import numpy as np
import matplotlib.pyplot as plot
from tensorflow.python.keras import backend as k
from keras import *
from keras.preprocessing.image import *
from keras.layers import *
from keras.callbacks import *
from flask import Flask,render_template,url_for,request
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import random
import os

def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = applications.ResNet50(weights=None, include_top=False,input_tensor=input_tensor)
    base_model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    final_output = Dense(n_out, activation='softmax', name='final_output')(x)
    model = Model(input_tensor, final_output)
    return model

def circular(img,sigmax):
    img=crop(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h,w,d=img.shape
    x=int(w/2)
    y=int(h/2)
    z=np.amin((x,y))
    circle_img=np.zeros((h,w),np.uint8)
    cv2.circle(circle_img,(x,y),int(z),1,thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop(img)
    img=cv2.addWeighted(img,4, cv2.GaussianBlur( img , (0,0) , sigmax) ,-3 ,128)
    return img;

def crop(img,tolerance=7):
    if img.ndim==2:
        mask=img>tolerance
        return img[np_ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        mask= gray_img>tolerance
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
    if (check_shape == 0): # image is too dark so that we crop out everything,
        return img # return original image
    else:
        img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
        img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
        img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
        # print(img1.shape,img2.shape,img3.shape)
        img = np.stack([img1,img2,img3],axis=-1)
        # print(img.shape)
    return img

def predict(img):
    canal =3
    height=width=320
    N_classess=5
   # tx = cv2.imread('stage2.jpeg')
    tx=circular(img, 20) 
    size = 320
    tx = cv2.resize(tx,(size,size))
    tx = np.array([tx]).astype('float')
    tx /= 255.0

    model = create_model(input_shape=( height,width, canal), n_out=N_classess)
    # model = load_model('model.h5')
    model.load_weights('DRmodel.h5')
    prediction = model.predict(tx)
    print(prediction)
    print(np.argmax(prediction))
    return np.argmax(prediction)

#predict()

app = Flask(__name__)

@app.route('/')

def xno():
  return render_template("test.html")



# @app.route('/upload')
# def upload_file():
#    return render_template('upload.html')
    
@app.route('/uploader', methods = ['POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files.getlist("file")
      path = "/Users/arunkrishna/Documents/project -DR-detection"
      print(f)
      results = []
      #f1 = request.files['right_image']
      for i in f:
      	filename = "uploaded"+str(random.randint(1,1000))+".jpeg"
      	print(filename)
      	i.save(os.path.join(path, filename))
      	print(i.filename)
      	img = cv2.imread(filename)
      	prediction = predict(img)
      	print(prediction)
      	results.append(prediction)
      	return results
      #return render_template("predicted.html", value = "prediction")
      #f1.save(secure_filename(f1.filename))

      

#@app.route('/')
#def index():
#  return render_template("xresult.html",value=i,f=image_to_test,path=pat)
#app.run(port=80,debug=True)
app.run(port=5677,debug=False, threaded=False)



