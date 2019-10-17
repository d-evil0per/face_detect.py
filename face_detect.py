import cv2
import tensorflow as tf
import os
import sys
import imutils
import numpy as np
import shutil
from PIL import Image
import random
import math

def banner():
    print("\n\n")
    print('#######    ###     ######  ########         ########  ######## ######## ########  ######  ######## ')
    print('##         ## ##   ##    ## ##               ##     ## ##          ##    ##       ##    ##    ##    ')
    print('##        ##   ##  ##       ##               ##     ## ##          ##    ##       ##          ##    ')
    print('######   ##     ## ##       ######           ##     ## ######      ##    ######   ##          ##    ')
    print('##       ######### ##       ##               ##     ## ##          ##    ##       ##          ##    ')
    print('##       ##     ## ##    ## ##               ##     ## ##          ##    ##       ##    ##    ##    ')
    print('##       ##     ##  ######  ######## ####### ########  ########    ##    ########  ######     ##    ')
    print("\n\n")
    

banner()
DATADIR = input("Face Recogination Model and Label Directory Path: ")
VIDEOTYPE=input("Video Type (WEBCAM | IP | FILE) : ")
DETECTIONTYPE=input("Face Detection (SINGLE | MULTI ) : ")






def convert_to_array(im):
    # im = cv2.imread(im)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((64, 64))
    return np.array(image)





def create_roi(frame,image_org,X,Y,W):
    ar=convert_to_array(frame)
    ar=ar/255
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=model.predict(a,verbose=1)
    # print(score)
    if DETECTIONTYPE=="SINGLE":
        label_index=np.argmax(score)
        acc=np.max(score)*100
        image=frame
        # y_cor=25
        text=""
        if acc>70:
            # R=random.randint(0,255)
            # G=random.randint(0,255)
            # B=random.randint(0,255)

            # text=CATEGORIES.item().get(int(label_index))+" : "+ str(round(acc))+"%"
            text=CATEGORIES.item().get(int(label_index)).split(" (")
            text=text[0]
            box_coords = ((X, Y-10), (W+120, Y - 40))
            cv2.rectangle(image_org, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
            cv2.rectangle(image_org, box_coords[0], box_coords[1], (255,255,255), 1)
            cv2.putText(image_org, text, (X+10, Y-20),  cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255), 1)
    elif DETECTIONTYPE=="MULTI":
        image=frame
        y_cor=Y-10
        text=""
        for i in range(len(score[0])):
            # print("loop............")
            # print(i)
            # print(CATEGORIES.item().get(int(i)))
            label_index=i
            if i>0:
                y_cor=y_cor+30
            else:
                y_cor=y_cor
            acc=np.max(score[0][i])*100
            # print(acc)
            if acc>60:
                # text=CATEGORIES.item().get(int(label_index))+" : "+ str(round(acc))+"%"
                text=CATEGORIES.item().get(int(label_index)).split(" (")
                text=text[0]
                box_coords = ((X, Y-10), (W+120, Y - 40))
                cv2.rectangle(image_org, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
                cv2.rectangle(image_org, box_coords[0], box_coords[1], (255,255,255), 1)
                cv2.putText(image_org, text, (X+10, Y-20),  cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255), 1)
    else:
        print("Invalid Detection Type!!")     
        

    cv2.imshow('Output',image_org)
    if VIDEOTYPE == "FILE":
        VIDEO_OUTPUT.write(image_org)






CATEGORIES=np.load(os.path.join(DATADIR,"categories.npy"),allow_pickle=True)
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('Assets/HAAR_CASCADES/haarcascade_frontalface_alt2.xml')

print(CATEGORIES)
execution_path = os.getcwd()

# cam = cv2.VideoCapture("http://192.168.0.100:4747/video")
if VIDEOTYPE=="WEBCAM":
    cam = cv2.VideoCapture(0)
elif VIDEOTYPE == "IPCAM":
    URL=input("IP Camera Address : ")
    cam = cv2.VideoCapture(URL)
elif VIDEOTYPE == "FILE":
    FILE=input("Video FILE Path : ")
    cam = cv2.VideoCapture(FILE)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    VIDEO_OUTPUT= cv2.VideoWriter(os.path.join(DATADIR,"OUTPUT_video.avi"),fourcc, 30.0, (1280,720))

cv2.namedWindow("Output",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output",1280,720)
IMG_SIZE = 64 
img_counter = 0
model = tf.keras.models.load_model(os.path.join(DATADIR,"Person_64x3_model.h5"))
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # print(faces)
    if len(faces)>0:
        for (x,y,w,h) in faces:
            roi_color = frame[y:y+h, x:x+w]
            cv2.line(frame,(x,y),(x+20,y),(255,255,255),2)
            cv2.line(frame,(x,y),(x,y+20),(255,255,255),2)
            cv2.line(frame,(x+w,y+h),(x+w-20,y+h),(255,255,255),2)
            cv2.line(frame,(x+w,y+h),(x+w,y+h-20),(255,255,255),2)
            cv2.line(frame,(x,y+h),(x+20,y+h),(255,255,255),2)
            cv2.line(frame,(x,y+h),(x,y+h-20),(255,255,255),2)
            cv2.line(frame,(x+w,y),(x+w-20,y),(255,255,255),2)
            cv2.line(frame,(x+w,y),(x+w,y+20),(255,255,255),2)
            create_roi(roi_color,frame,x,y,x+w)
            img_counter += 1
    else:
        cv2.imshow('Output',frame)
        if VIDEOTYPE == "FILE":
            VIDEO_OUTPUT.write(frame)

    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        print("Escape hit, closing...")
        break
    
cam.release()
cv2.destroyAllWindows()


    






'''
Nikolaj Coster-Waldau =Jaime Lannister
Lena Headey =Cersei Lannister
Emilia Clarke=Daenerys Targaryen
Iain Glen=Jorah Mormont
Kit Harington=Jon Snow
Sophie Turner=  Sansa Stark
Maisie Williams=Arya Stark
Alfie Allen =Theon Greyjoy
Isaac Hempstead Wright= Bran Stark
Jack Gleeson=Joffrey Baratheon
Rory McCann=The Hound
Peter Dinklage=Tyrion Lannister
Jason Momoa=    Khal Drogo
Aidan Gillen=Littlefinger
John Bradley=Samwell Tarly
Sean Bean =Eddard Ned Stark
Michelle Fairley=Catelyn Stark
'''