# Facial Recogination

Face Recogination From the Scratch From Creating Image DataSet to Training and Evaluating the Model. 


# Requirements

- Google Image Downloader Library- **pip3 install google_images_download**
- Chrome Web Driver-**[ChromeDriver - WebDriver for Chrome](https://sites.google.com/a/chromium.org/chromedriver/)**
- OpenCV- **pip3 install opencv-python**
- TensorFlow- **pip3 install tensorflow**
- Keras- **pip3 install keras**
- Numpy- **pip3 install numpy**
- Pillow- **pip3 install pillow**
- imutils- **pip3 install imutils**
- Shutil- **pip3 install pytest-shutil**
- Pathlib- **pip3 install pathlib**
- imghdr- **pip3 install micropython-imghdr**

## Step 1 : Creating Image DataSet using Google Image Downloader
>### **python3 imgdownloader.py** 
- For example : I want to build a face recogination for All Game of Characters Like:
	- Nikolaj Coster-Waldau =Jaime Lannister
	- Lena Headey =Cersei Lannister
	- Emilia Clarke=Daenerys Targaryen
	- Iain Glen=Jorah Mormont
	- Kit Harington=Jon Snow
	- Sophie Turner=  Sansa Stark
	- Maisie Williams=Arya Stark
	- Alfie Allen =Theon Greyjoy
	- Isaac Hempstead Wright= Bran Stark
	- Jack Gleeson=Joffrey Baratheon
	- Rory McCann=The Hound
	- Peter Dinklage=Tyrion Lannister
	- Jason Momoa=    Khal Drogo
	- Aidan Gillen=Littlefinger
	- John Bradley=Samwell Tarly
	- Sean Bean =Eddard Ned Stark
	- Michelle Fairley=Catelyn Stark
- We need to Download lots of images of these character and to do that we will run **"imgdownloader file"**
	- It will ask for user inputs like
		-  Enter any keyword or keywords(sperated by comma) **for eg. Nikolaj Coster-Waldau,Lena Headey,Emilia Clarke**
		- Number of Image for each keywords **for eg. 2000** for 2000 images for each
		- Output Folder name **Assets/Sample/GOT**. full path of the folder where you want to create an output folder 
	- More the number of image more time it will take. **Note: Number of image and number of downloaded image may vary due to lack of image or due to some error.**
	- once Downloads Completed You will find the output folder inside the location you have provided and subfolders with each keyword name inside the output folder. 
		- **for eg. Assets/Example/GOT/Nikolaj Coster-Waldau**
		- **for eg. Assets/Example/GOT/Lena Headey**
		- **for eg. Assets/Example/GOT/Emilia Clarke**


## Step 2:  Data cleansing from the image dataset
> ### python3 faceExtrator.py
- Downloaded Image may or may not contain **Relevant face**, So we need to **Clean** it to make our **Model More accurate and effecient**. To do so.
	- It requires an input which is nothing but the path of the Directory which consist the Dataset for eg. **Assets/Example/GOT**
	- After that it Will Transverse through each Image file in the Dataset folder i.e **Assets/Example/GOT**
	- Then it will Convert the image in **Black & White** , it will Detect a Face and Get the **Region Of Interest (Roi)** and Save it as new image in the Output folder after Resizing it to **64X64 ( dimension can be change but it should be small )**
- After Cleansing we will get the more Accurate data to extract features from it. 

## Step 3 : Training and Evaluation
> ### python3 modelt&p.py

- First the code will **Extract the Labels and Feature** from the Image and Save it in a Folder inside the Example Folder.
- Then it will **Save Labels and Features in a File**.
- Then it will **Shuffles** the data and Decide the **Percentage of Training** and Testing Dataset.
- After that it will Start its **Training** and  generate a stuitable **classifier** for the **face recogination.**
- It will Save the **Classifier** in the Output Folder.


## Final Step: Evaluation and Detection
> ### python3  face_detect.py
- It has Variety of options Like
	- Detect Through Webcam 
	- Detect Through IP Camera
	- Detect Using Video File
- It supports Single Face detection and multiple face detection as well
- Once Face is Detected it will Preddict Name the Person 
    