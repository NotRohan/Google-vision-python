import cv2
import numpy as np
import io
import os
from google.cloud import vision
from google.cloud.vision import types

cap = cv2.VideoCapture("a.mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
currentFrame = 0
while(currentFrame < length): # making frames from video
    ret, frame = cap.read()
    name = str(currentFrame) + '.jpg'
    print ('Creating frame : ' + name)
    cv2.imwrite('images/'+name,frame)
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
    currentFrame += 60
    cap.set(1,currentFrame)

cap.release()
cv2.destroyAllWindows()