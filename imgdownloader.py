import sys,os
from google_images_download import google_images_download
from datetime import datetime


def download(Files,Limit,FolderName,Driver="/usr/lib/chromium-browser/chromedriver",PRINTURL=False):
    print("Downloading ["+str(Files)+"] From Google" )
    # datetime object containing current date and time
    now = datetime.now()
    print("Downloading Started at "+now.strftime("%d/%m/%Y %H:%M:%S"))
    response = google_images_download.googleimagesdownload()   #class instantiation
    arguments = {"keywords":Files,"limit":Limit,"print_urls":PRINTURL,"chromedriver":"/usr/local/lib/node_modules/webdriver-manager/selenium/chromedriver_77.0.3865.40"}   #creating list of arguments 
    response.download(arguments)
    now = datetime.now()
    print("Downloading ended at "+now.strftime("%d/%m/%Y %H:%M:%S"))
    os.rename('downloads',str(FolderName)) 


def collectinfo():
    dirname, filename = os.path.split(os.path.abspath(__file__))
    os. chdir(dirname)
    Files=input("Enter any keyword or keywords(sperated by comma): ")
    Limit=input("Number of Image for each keywords: ")
    FolderName=input("Output Folder name: ")
    PRINTURL=input("Do you want to Print urls? (y/n Default=n)  OR Press ENTER to skip : ")
    if PRINTURL.lower()=="y":
        PRINTURL=True
    else:
        PRINTURL=False

    Driver=input("Chrome Driver's Path  (Default Path='/usr/lib/chromium-browser/chromedriver') Press ENTER to skip : ")
    download(Files,Limit,FolderName,Driver,PRINTURL)

def banner():
    print("\n\n")
    print('#### ##     ##  ######   ########   #######  ##      ## ##    ## ##        #######     ###    ########  ######## ########  ')
    print(' ##  ###   ### ##    ##  ##     ## ##     ## ##  ##  ## ###   ## ##       ##     ##   ## ##   ##     ## ##       ##     ## ')
    print(' ##  #### #### ##        ##     ## ##     ## ##  ##  ## ####  ## ##       ##     ##  ##   ##  ##     ## ##       ##     ## ')
    print(' ##  ## ### ## ##   #### ##     ## ##     ## ##  ##  ## ## ## ## ##       ##     ## ##     ## ##     ## ######   ########  ')
    print(' ##  ##     ## ##    ##  ##     ## ##     ## ##  ##  ## ##  #### ##       ##     ## ######### ##     ## ##       ##   ##   ')
    print(' ##  ##     ## ##    ##  ##     ## ##     ## ##  ##  ## ##   ### ##       ##     ## ##     ## ##     ## ##       ##    ##  ')
    print('#### ##     ##  ######   ########   #######   ###  ###  ##    ## ########  #######  ##     ## ########  ######## ##     ## ')
    print("\n\n")
    collectinfo()

banner()
