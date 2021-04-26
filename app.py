import cv2
import numpy as np
import matplotlib.pyplot as plot

from keras import *
from keras.preprocessing.image import *
from keras.layers import *
from keras.callbacks import *
from flask import Flask,render_template,url_for,request,make_response
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
import pdfkit
import os
import pdfkit
from PIL import Image
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
from flask import send_from_directory

# img1="";img2="";plot1="";plot2="";dgsdg
path = ""

X=[]
val=[]
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
    # imshow(tx)
    tx = np.array([tx]).astype('float')
    tx /= 255.0

    model = create_model(input_shape=( height,width, canal), n_out=N_classess)
    # model = load_model('model.h5')
    model.load_weights('model.h5')
    prediction = model.predict(tx)
    print(prediction)
    print(np.argmax(prediction))
    pred=np.argmax(prediction)
    return prediction,pred

#predict()

app = Flask(__name__)


@app.route('/')
def xno():
  return render_template("test.html")


@app.route('/uploader', methods = ['POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files.getlist("file")
     
      results = []
     
      # img1="";img2="";plot1="";plot2="";
      #f1 = request.files['right_image']
      for i in range(2):
        file_base = "uploaded"+str(random.randint(1,1000))
        filename = file_base+".jpeg"
        print(filename)
        # f[i].save(os.path.join(path,filename))
        f[i].save(os.path.join(os.path.join(path,"static"), filename))
       
       # f[i].save(os.path.join(path, filename))

        print(f[i].filename)
        img = cv2.imread(os.path.join(os.path.join(path,"static"), filename))
        # prediction = predict(img)
        x,pre=predict(img)
        x=x.flatten()
        y=[0,1,2,3,4]
        plot.ylabel('PROBABILITY')
        plot.xlabel('STAGE')
        plot.bar(y,x,color='green', edgecolor = 'red')      #actual stage 2
        
        plot_name = "plot"+file_base+".jpeg"
        plot.savefig(os.path.join(os.path.join(path,"static"), plot_name))
        plot.savefig(os.path.join(path,plot_name))
        if i==0:
          img1=filename;plot1=plot_name;X.append(img1);X.append(plot1)
        else:
          img2=filename;plot2=plot_name; X.append(img2);X.append(plot2)
        # print(prediction)
        plot.close()
        ab=plot.show()
        plot.close()
        plot.close()


        # plot_name=""
        results.append(x)
        val.append(pre)
        
        print(os.path.join(os.path.join(path,"static"), str(img1)))
      return render_template("dr.html", im1= img1, im2=img2, plt1=plot1, plt2=plot2,res= val)
      #return render_template("predicted.html", value = "prediction")
      #f1.save(secure_filename(f1.filename))

      

@app.route('/download', methods=['GET','POST'])
def downloadPdf():
    output=BytesIO()
    canvas = canvas.Canvas(output, pagesize=letter)
    width,height = letter
    # print(width)
    # print(height)
    canvas.setLineWidth(.3)
    canvas.setFont('Helvetica', 20)
    canvas.drawString(width/2-60,750,'MEDICAL REPORT')

    canvas.drawImage( path+"static/"+str(X[0]), 50,500, 3*inch,3*inch) 
    canvas.drawImage( path+"static/"+str(X[2]), width-50-(3*inch),500, 3*inch,3*inch) 
    canvas.drawImage( path+"static/"+str(X[1]), 50,500-3*inch-50, 3*inch,3*inch)       
    canvas.drawImage(path+"static/"+str(X[3]), width-50-(3*inch),500-3*inch-50, 3*inch,3*inch) 
    canvas.drawString(100,500-3*inch-150,"LEFT EYE STAGE: "+str(val[0]))
    canvas.drawString(width-(2*inch)-100,500-3*inch-150,"RIGHT EYE STAGE: "+str(val[1]))  

    canvas.save()
    pdf_out = output.getvalue()
    output.close()

    response = make_response(pdf_out)
    response.headers['Content-Disposition'] = "attachment; filename='REPORT1.pdf"
    response.mimetype = 'application/pdf'
    return response

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                          'favicon.ico',mimetype='image/vnd.microsoft.icon')
  #comment
if __name__ == '__main__':
    app.run(debug=False, threaded = False)



