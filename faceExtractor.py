#!/usr/bin/python
from PIL import Image
import os
import sys
from pathlib import Path
import imghdr

import numpy as np
import cv2 
from datetime import datetime


face_cascade = cv2.CascadeClassifier('Assets/HAAR_CASCADES/haarcascade_frontalface_default.xml')

# cv2.namedWindow("Output",cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Output",600,800)

def convert_to_array(img):
    # im = cv2.imread(im)
    # img = Image.fromarray(im, 'RGB')
    image = img.resize((64, 64))
    return np.array(image)

def resize(file,folder,count,OUTPUT):
	type_img=imghdr.what(file)
	ext = ["jpg","jpeg","png","gif","bmp"] 
	if type_img in ext:
		# print(file)
		try:
			image = Image.open(file).convert('RGB')
			# print(image)
			opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
			gray = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)
			j=0
			for (x,y,w,h) in faces:
				j=j+1
				roi_gray = gray[y:y+h, x:x+w]
				im = Image.fromarray(roi_gray)
				ar=convert_to_array(im)
				FOLDER=os.path.join(OUTPUT,folder)
				if not os.path.exists(FOLDER):
					os.makedirs(FOLDER)
				output_file_name = os.path.join(FOLDER, os.path.basename(folder)+"_"+str(count)+"_"+str(j)+".jpg")
				print(output_file_name)
				cv2.imwrite(output_file_name,ar)
				# ar.save(output_file_name, "JPEG", quality = 95)

				continue
		except:
			print("Something went wrong when Processing the file")



def recur(folder_path,OUTPUT):
	p=Path(folder_path)
	dirs=p.glob("*")
	i=0
	for folder in dirs:
		print(folder)
		if folder.is_dir():
			recur(folder,OUTPUT)
		else:
			i+=1
			resize(folder,folder_path,i,OUTPUT)
			


def banner():
	print("\n\n")
	print('########    ###     ######  ######## ######## ##     ## ######## ########     ###     ######  ########  #######  ########  ')
	print('##         ## ##   ##    ## ##       ##        ##   ##     ##    ##     ##   ## ##   ##    ##    ##    ##     ## ##     ## ')
	print('##        ##   ##  ##       ##       ##         ## ##      ##    ##     ##  ##   ##  ##          ##    ##     ## ##     ## ')
	print('######   ##     ## ##       ######   ######      ###       ##    ########  ##     ## ##          ##    ##     ## ########  ')
	print('##       ######### ##       ##       ##         ## ##      ##    ##   ##   ######### ##          ##    ##     ## ##   ##   ')
	print('##       ##     ## ##    ## ##       ##        ##   ##     ##    ##    ##  ##     ## ##    ##    ##    ##     ## ##    ##  ')
	print('##       ##     ##  ######  ######## ######## ##     ##    ##    ##     ## ##     ##  ######     ##     #######  ##     ## ')
	print("\n\n")
	directory = input("Enter Image folder path to extract Faces :")
	dirname, filename = os.path.split(os.path.abspath(__file__))
	os. chdir(dirname)
	OUTPUT=os.path.join(dirname,str(directory)+"_Faces")
	# datetime object containing current date and time
	now = datetime.now()
	print("Face Extraction Started at "+now.strftime("%d/%m/%Y %H:%M:%S"))
	recur(directory,OUTPUT)
	# datetime object containing current date and time
	now = datetime.now()
	print("Face Extraction ended at "+now.strftime("%d/%m/%Y %H:%M:%S"))



banner()

