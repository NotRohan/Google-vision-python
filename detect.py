import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import io
import os
from google.cloud import vision
from google.cloud.vision import types
from PIL import ImageFilter, Image
import time


cap = cv2.VideoCapture("vid.VOB")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
currentFrame = 0
while(currentFrame < length): # making frames from video
    ret, frame = cap.read()
    name = str(currentFrame) + '.jpg'
    print ('Creating frame : ' + name)
    im = frame[0:20,0:290]   # cropping frame
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # converting rgb to gray
    im_gray = cv2.bitwise_not(im_gray)
    im_bw = cv2.threshold(im_gray,160,255,cv2.THRESH_TRUNC)[1] # further image preprocessing
    cv2.imwrite('images/'+name,im_bw)
    # google vision start
    client = vision.ImageAnnotatorClient()

    file_name = os.path.join(
        os.path.dirname(__file__),
        'images/'+name)

    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    response = client.text_detection(image=image) #api hit
    labels = response.text_annotations
    print('----------start----------')   
    for label in labels:
        print("label : "+label.description) # print all the labels in image
    print('----------end----------\n')
    currentFrame += 25
    cap.set(1,currentFrame)

cap.release()
cv2.destroyAllWindows()