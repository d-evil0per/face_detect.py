from PIL import Image
import numpy as np
import os
import cv2
import imghdr
import sys
from pathlib import Path

import keras
from keras.utils import np_utils
# import sequential model and all the required layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from keras.models import Model, load_model
from keras import backend as K


#extracting label & features 




global folder_count,x_train,y_train,x_test,y_test
data=[]
labels=[]
categories={}



def data_shuffling(OUTPUT):
    print("Initializing Data Shuffling....")
    person=np.load(os.path.join(OUTPUT,"person.npy"))
    labels=np.load(os.path.join(OUTPUT,"labels.npy"))
    s=np.arange(person.shape[0])
    np.random.shuffle(s)
    person=person[s]
    labels=labels[s]
    num_classes=len(np.unique(labels))
    data_length=len(person)

    (x_train,x_test)=person[(int)(0.1*data_length):],person[:(int)(0.1*data_length)]
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    train_length=len(x_train)
    test_length=len(x_test)

    (y_train,y_test)=labels[(int)(0.1*data_length):],labels[:(int)(0.1*data_length)]
    #One hot encoding
    y_train=keras.utils.to_categorical(y_train,num_classes)
    y_test=keras.utils.to_categorical(y_test,num_classes)
    print("Data Shuffling has been Completed....")
    make_compile_modal(x_train,y_train,x_test,y_test)
    
def make_compile_modal(x_train,y_train,x_test,y_test):
    print("Initializing Model Creation and Training Process.... ")
    #make model
    model=Sequential()
    model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(64,64,3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(500,activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(len (categories),activation="sigmoid"))
    model.summary()

    print("Modal Compilation started.....")
    # compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Model Compilation Completed ....")
    print("Model Training Intialized....")
    model.fit(x_train,y_train,batch_size=50,epochs=20,verbose=1)
    print("Model Training Completed....")
    print("Model Evaluation Intialized....")
    score = model.evaluate(x_test, y_test, verbose=1)
    print("Model Evaluation Completed....")
    print('\n', 'Test accuracy:', score[1])
    # save modal
    print("Saving Model....")
    model.save(os.path.join(OUTPUT,"Person_64x3_model.h5"))
    print("Model Saved Successfully....")
    predict_person(TESTIMG,model)


def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((64, 64))
    return np.array(image)
def get_person_name(label):
    return categories[label]


def predict_person(file,model):
    print("Predicting .................................")
    ar=convert_to_array(file)
    ar=ar/255
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=model.predict(a,verbose=1)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    person=get_person_name(label_index)
    print(person)
    print("The predicted Person is a "+person+" with accuracy =    "+str(acc))

def banner():
    print("\n\n")
    print('##     ##  #######  ########  ######## ##       ########   ####    ########  ')
    print('###   ### ##     ## ##     ## ##       ##          ##     ##  ##   ##     ## ')
    print('#### #### ##     ## ##     ## ##       ##          ##      ####    ##     ## ')
    print('## ### ## ##     ## ##     ## ######   ##          ##     ####     ########  ')
    print('##     ## ##     ## ##     ## ##       ##          ##    ##  ## ## ##        ')
    print('##     ## ##     ## ##     ## ##       ##          ##    ##   ##   ##       ') 
    print('##     ##  #######  ########  ######## ########    ##     ####  ## ##       ') 
    print("\n\n")

def label_and_feature(file,folder,i):
    type_img=imghdr.what(file)
    ext = ["jpg","jpeg","png","gif","bmp"] 
    if type_img in ext:
        imag=cv2.imread(str(file))
        data.append(np.array(imag))
        labels.append(i)
        categories[i]=os.path.basename(folder)


        


def recur(folder_path,folder_count):
    p=Path(folder_path)
    dirs=p.glob("*")
    # print(dirs)
   
    for folder in dirs:
        # print(folder)
        if folder.is_dir():
            recur(folder,folder_count)
            folder_count+=1
        else:
            
            label_and_feature(folder,folder_path,folder_count)

def init_labeling(DATADIR): #1st step
    print("Initializing Labeling....")
    folder_count=0
    recur(DATADIR,folder_count)
    print("Labels & Features Has been Extracted... ")

def save_labels(OUTPUT,data,labels,categories):
    print("Saving Labels and Features....")
    person=np.array(data)
    labels=np.array(labels)
    np.save(os.path.join(OUTPUT,"person"),person)
    np.save(os.path.join(OUTPUT,"labels"),labels)
    np.save(os.path.join(OUTPUT,"categories"),categories)    
    print("Labels and Features has been Saved.")


banner()
DATADIR = input("Enter path of Image dataset: ")
TESTIMG = input("Enter the path of Test Image: ")
OUTPUT = input("Enter the path for OUPUT DATA: ")
if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)
init_labeling(DATADIR)
save_labels(OUTPUT,data,labels,categories)
data_shuffling(OUTPUT)

