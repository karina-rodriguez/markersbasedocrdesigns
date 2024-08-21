#https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
#import the necessary packages
#check for id to be the first top left and top right
#save image color format and add to csv the image file.
import argparse
import imutils
import cv2
import sys
import os
from random import randint, randrange, seed

from ocrmac import ocrmac
#from PIL import Image
import PIL.Image
from PIL.ExifTags import TAGS
from pprint import pprint
from os import listdir
import numpy as np
import datetime
from csv import writer
from pathlib import Path
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import data
from skimage import transform
from skimage import io
import sys

def display_image(imagetodisplay):
    #images = "BT53-209/IMG_20240724_154908~2.jpg"
    fig, ax = plt.subplots()
    # image = mpimg.imread(imagetodisplay)
    # new_frame = Image.open(imgs[count])

    im = ax.imshow(imagetodisplay, interpolation='none')
    plt.show()  

def sort_by_X(annotations):
    array_x = []
    for i in range(len(annotations)):
        array_x.insert(i, annotations[i][2][0])
    
    #print(array_x)

    for i in range(len(annotations)):
        #print("start")

        #print(annotations[i:])
        #print(annotations[i][1][0])
        swap = i + np.argmin(array_x[i:])
        #print(i + np.argmin(annotations[i:]))
        #print("swap ",swap)
        (array_x[i], array_x[swap]) = (array_x[swap], array_x[i])
        (annotations[i], annotations[swap]) = (annotations[swap], annotations[i])
        #print("array_x ",array_x)
        #print("result ",annotations)
    return annotations

def sort_by_Y(annotations):
    array_y = []
    for i in range(len(annotations)):
        array_y.insert(i, annotations[i][2][1])
    
    #print(array_y)

    for i in range(len(annotations)):
        #print("start")

        #print(annotations[i:])
        #print(annotations[i][1][0])
        swap = i + np.argmax(array_y[i:])
        #print(i + np.argmin(annotations[i:]))
        #print("swap ",swap)
        (array_y[i], array_y[swap]) = (array_y[swap], array_y[i])
        (annotations[i], annotations[swap]) = (annotations[swap], annotations[i])
        #print("array_x ",array_x)
    return annotations
def find_line_y_for_text(text,annotations):    
    for i in range(len(annotations)):
        textinline = annotations[i][0]
        #print(linenum)
        #print()
        if text == textinline:
            #valuey is the initial y plus the height
            valuey = annotations[i][2][1]+annotations[i][2][3]+0.02

            return valuey

def find_line(line,annotations):    

    for i in range(len(annotations)):
        linenum = annotations[i][2][0]
        #print(linenum)
        #print()
        if (line-linenum)<0.03:
            return i

def find_text_between_X_values(xmin,xmax,annotations):    
    newarray = []
    for i in range(len(annotations)):
        #print("***",annotations[i][2])
        x = annotations[i][2][0]
        #print(float(x)>xmin)
        #print(float(x)<xmax)
        if float(x)>xmin and float(x)<xmax:
            newarray.insert(i,annotations[i])
        
    return newarray

def find_text_between_Y_values(ymin,ymax,annotations):    
    newarray = []
    for i in range(len(annotations)):
        y = annotations[i][2][1]
        if float(y)>ymin and float(y)<ymax:
            newarray.insert(i,annotations[i])
        
    return newarray


def first_approach_read(imagecropped,newpath,
filename,camera,ordered,c):
    alldataarray = []
    count =0

    alldataarray.insert(count,c)
    count +=  1

    #***************************************************************
    #**********************get design no**********************
    #***************************************************************
    #designnumarray = find_text_between_Y_values(0,0.1,ordered)   
    designnumarray = find_text_between_Y_values(0.87,1,ordered)   
    #print(len(designnumarray))
    ''' this is the old way which was finding the first 
    two values but it's no robust enoguh
    designno = [ordered[0],ordered[1]]
    '''
    #sort them by Y so we get the bottom part of the image or headers first
    designno_byY = sort_by_X(designnumarray)
    print("\nDesign No.: ",designno_byY)
    #create a word iterating over all values in the array
    worddesignnumadd = ""
    for worddesignnumaddi in range(0,len(designno_byY)):
        worddesignnumadd += designno_byY[worddesignnumaddi][0]
    worddesignnumadd = worddesignnumadd.replace('Registered Design No','')
    worddesignnumadd = worddesignnumadd.replace('.','')
    print(worddesignnumadd)
    alldataarray.insert(count,worddesignnumadd)
    count += 1
    
    #***************************************************************
    #**********************get date application**********************
    #***************************************************************
    #dateapparray = find_text_between_Y_values(0.1,0.16,ordered)   
    dateapparray = find_text_between_Y_values(0.82,0.87,ordered)   
    #print(len(dateapparray))
    ''' this is the old way which was finding the  
    two values but it's no robust enoguh
    dateapp = [ordered[2],ordered[3]]
    '''
    #sort them by Y so we get the bottom part of the image or headers first
    dateapp_byY = sort_by_Y(dateapparray)
    print("\nDate app: ",dateapp_byY)
    #create a word iterating over all values in the array
    worddateappadd = ""
    for worddateappaddi in range(0,len(dateapp_byY)):
        worddateappadd += dateapp_byY[worddateappaddi][0]
    worddateappadd = worddateappadd.replace('Date of Application','')
    print(worddateappadd)
    alldataarray.insert(count,worddateappadd)
    count += 1

    #***************************************************************
    #**********************get date reg**********************
    #***************************************************************
    #dateregarray = find_text_between_Y_values(0.16,0.22,ordered)   
    dateregarray = find_text_between_Y_values(0.79,0.82,ordered)   
    #print(len(dateregarray))
    ''' this is the old way which was finding the  
    two values but it's no robust enoguh
    datereg = [ordered[4],ordered[5]]
    '''
    #sort them by Y so we get the bottom part of the image or headers first
    datereg_byY = sort_by_Y(dateregarray)
    print("\nDate reg: ",datereg_byY)
    #create a word iterating over all values in the array
    worddateregadd = ""
    for worddateregaddi in range(0,len(datereg_byY)):
        worddateregadd += datereg_byY[worddateregaddi][0]
    worddateregadd = worddateregadd.replace('Date as of which design registered','')
    print(worddateregadd)
    alldataarray.insert(count,worddateregadd)
    count += 1

    #***************************************************************
    #**********************get date cert**********************
    #***************************************************************
    #datecerarray = find_text_between_Y_values(0.22,0.28,ordered)  
    datecerarray = find_text_between_Y_values(0.75,0.79 ,ordered)    
    #print(len(datecerarray))
    ''' this is the old way which was finding the  
    two values but it's no robust enoguh
    datecert = [ordered[6],ordered[7]]
    '''
    #sort them by Y so we get the bottom part of the image or headers first
    datecer_byY = sort_by_Y(datecerarray)
    print("\nDate cer: ",datecer_byY)
    #create a word iterating over all values in the array
    worddateceradd = ""
    for worddateceraddi in range(0,len(datecer_byY)):
        worddateceradd += datecer_byY[worddateceraddi][0]
    worddateceradd = worddateceradd.replace('Certificate of registration issued','')
    print(worddateceradd)
    alldataarray.insert(count,worddateceradd)
    count += 1

    #***************************************************************
    #**********************get article**********************
    #***************************************************************
    #artarray = find_text_between_Y_values(0.28,0.47,ordered)
    artarray = find_text_between_Y_values(0.65,0.75,ordered)   
    #print(len(artarray))
    ''' this is the old way which was finding the  article
    #now get article
    i=8
    keepgoing=1
    art=[]
    linex = ordered[i][2][0]
    newlinex = ordered[i][2][0]
    while keepgoing==1:
        text = ordered[i][0]
        #print(text)
        texttofind = "Name and address"
        if texttofind.lower() in text.lower():
            keepgoing = 0
        else:                 
            newlinex = ordered[i][2][0]
            #check space to ensure it is in the same are
            if (newlinex-linex<0.09):
                art.insert(0,ordered[i])
        if keepgoing==0:
            break
        i=i+1

    #get art
    art_byY = sort_by_Y(art)
    #print("\nArticle: ",art_byY)
    wordart = ""
    #print(len(art_byY))
    '''
    #sort them by Y so we get the bottom part of the image or headers first
    art_byY = artarray#sort_by_Y(artarray)
    print("\nArticle: ",art_byY)
    #create a word iterating over all values in the array
    wordartadd = ""
    for wordartaddi in range(0,len(art_byY)):
        wordartadd += art_byY[wordartaddi][0]+" "
    wordartadd = wordartadd.replace('Article in respect of which design is registered','')
    print(wordartadd)
    alldataarray.insert(count,wordartadd)
    count += 1

    #***************************************************************
    #**********************get proprietor**********************
    #***************************************************************
    #proparray = find_text_between_Y_values(0.47,0.65,ordered) 
    perdesign = find_line_y_for_text("Address for Service",ordered)    
    #print ("1",perdesign)
    if perdesign == None:
        perdesign = 0.35
    #print ("2",perdesign)
    proparray = find_text_between_Y_values(perdesign,0.65,ordered)  
    #print(len(proparray))    
    #sort them by Y so we get the bottom part of the image or headers first
    prop_byY = proparray#sort_by_Y(artarray)
    print("\nProprietary: ",prop_byY)
    #create a word iterating over all values in the array
    wordpropadd = ""
    for wordpropaddi in range(0,len(prop_byY)):
        wordpropadd += prop_byY[wordpropaddi][0]+" "
    wordpropadd = wordpropadd.replace('Name and address of proprietor','')
    print(wordpropadd)
    alldataarray.insert(count,wordpropadd)
    count += 1

    #***************************************************************
    #**********************get address for service**********************
    #***************************************************************
    #addserarray = find_text_between_Y_values(0.65,0.81,ordered) 
    addserarray = find_text_between_Y_values(perdesign-0.2,perdesign,ordered)     
    #print(len(addserarray))    
    #sort them by Y so we get the bottom part of the image or headers first
    addser_byY = addserarray#sort_by_Y(artarray)
    print("\nAddress service: ",addser_byY)
    #create a word iterating over all values in the array
    wordaddseradd = ""
    for wordaddseraddi in range(0,len(addser_byY)):
        wordaddseradd += addser_byY[wordaddseraddi][0]+" "
    wordaddseradd = wordaddseradd.replace('Address for Service','')
    print(wordaddseradd)
    alldataarray.insert(count,wordaddseradd)
    count += 1

    #***************************************************************
    #**********************any other notes**********************
    #***************************************************************
    onotesarray = find_text_between_Y_values(0,perdesign-0.2,ordered)   
    print(len(onotesarray))    
    #sort them by Y so we get the bottom part of the image or headers first
    onotes_byY = onotesarray#sort_by_Y(artarray)
    print("\nOther notes: ",onotes_byY)
    #create a word iterating over all values in the array
    wordonotesadd = ""
    for wordonotesaddi in range(0,len(onotes_byY)):
        wordonotesadd += onotes_byY[wordonotesaddi][0]+" "
    print(wordonotesadd)
    alldataarray.insert(count,wordonotesadd)
    count += 1


    

    alldataarray.insert(count,datetime.datetime.now().strftime("%d/%B/%Y"))
    count += 1
    alldataarray.insert(count,"Izzy Barrett-Lally and Alfie Lien-Talks")
    count += 1
    alldataarray.insert(count,camera)
    count += 1
    alldataarray.insert(count,filename)
    count += 1
    #print("annotations id",id)
    fullnamefile = str(worddesignnumadd)+"_"+str(c)+".jpg" 
    cv2.imwrite(newpath+"/"+fullnamefile,imagecropped)
    alldataarray.insert(count,newpath+"/"+fullnamefile)
    count +=  1     
    alldataarray.insert(count,"BT53-209")
    return alldataarray
    

def write_to_csv(csvfile,data):
    with open(csvfile, 'a') as f_object:
        writer_object = writer(f_object)
        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(datatoadd)

        # Close the file object
        f_object.close()

def find_id(annotations):
    for i in range(len(annotations)):
        print("-----my:"+str(annotations[i][2][2]))
        if annotations[i][2][2]>0.12 and annotations[i][2][2]<0.3:
            print( annotations[i][0])
            return annotations[i][0]
        

def get_data(imagetodisplay,imagecropped,newfolder,c):
    alldataarray = []
    count =0

    alldataarray.insert(count,c)
    count +=  1    
    image = PIL.Image.fromarray(imagecropped)
    # Get the exif data and map to the correct tags
    # exif_data = {
    #                 PIL.ExifTags.TAGS[k]: v
    #                 for k,v in image._getexif().items()
    #                 if k in PIL.ExifTags.TAGS
    #             }
    # #pprint(exif_data)
    annotations = ocrmac.OCR(image).recognize()
    annotations_ordered = sort_by_Y(annotations)
    # print("all annotations",annotations_ordered)
    #print("annotations ",annotations)
    id = find_id(annotations_ordered)
    alldataarray.insert(count,id)
    count +=  1 

    #print("annotations id",id)
    fullnamefile = str(id)+"_"+str(c)+".jpg" 
    cv2.imwrite(newpath+"/"+fullnamefile,imagecropped)
    alldataarray.insert(count,fullnamefile)
    count +=  1 

    return alldataarray
    # display_image(image)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,
	help="path to input folder containing images")
ap.add_argument("-t", "--type", type=str,
	default="DICT_ARUCO_ORIGINAL",
	help="type of ArUCo tag to detect")
ap.add_argument("-s", "--seed", type=int,
	default="456789",
	help="seed number")
ap.add_argument("-a", "--archive", type=str,
	default="",
	help="archive code")
args = vars(ap.parse_args())

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


# load the input image from disk and resize it
print("[INFO] loading folder..."+args["folder"])
print(args["folder"])
folder_dir = args["folder"]
theseed = args["seed"]
archive_code = args["archive"]
newpath  = folder_dir+'_cropped'
if not os.path.exists(newpath):
    os.makedirs(newpath)
c=0
seed(theseed)

for images in os.listdir(folder_dir):
    print("kre")
    print(images)
    if (images.endswith(".jpeg")):
        print(images)
        # images to add
        #imagetodisplay = folder_dir+"/"+"IMG_20240724_135949~2.jpg"#images
        #display_image(imagetodisplay)
        image = cv2.imread(folder_dir+"/"+images)
        image = imutils.resize(image, width=800)

        #image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #metadata
        imageme = PIL.Image.open(folder_dir+"/"+images)
        # Get the exif data and map to the correct tags
        exif_data = {
                        PIL.ExifTags.TAGS[k]: v
                        for k,v in imageme._getexif().items()
                        if k in PIL.ExifTags.TAGS
                    }

        # verify that the supplied ArUCo tag exists and is supported by
        # OpenCV
        if ARUCO_DICT.get(args["type"], None) is None:
            print("[INFO] ArUCo tag of '{}' is not supported".format(
                args["type"]))
            sys.exit(0)
        # load the ArUCo dictionary, grab the ArUCo parameters, and detect
        # the markers
        print("[INFO] detecting '{}' tags...".format(args["type"]))
        arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
            parameters=arucoParams)
        print(len(corners))
        topLeftX = topRightX = bottomRightX = bottomLeftX = 0
        topLeftY = topRightY = bottomRightY = bottomLeftY = 0
        counter = 0 
        # verify we have the four markers detected
        if len(corners) >1:
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):

            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                # draw the bounding box of the ArUCo detection
                # cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                # cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                # cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                # cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
                # compute and draw the center (x, y)-coordinates of the ArUco
                # marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)

                # print(markerID)
                if markerID == 15:
                    topLeftX=bottomRight[0]
                    topLeftY=bottomRight[1]
                elif(markerID == 5):
                    topRightX=cX
                    topRightY=cY
                elif(markerID == 20):
                    bottomLeftX=cX
                    bottomLeftY=cY
                elif(markerID == 10):
                    bottomRightX=topLeft[0]
                    bottomRightY=topLeft[1]


                # print("centre: ("+str(cX)+","+str(cY)+")")
                # cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                # # draw the ArUco marker ID on the image
                # cv2.putText(image, str(markerID),
                # 	(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                # 	0.5, (0, 255, 0), 2)
                # print("[INFO] ArUco marker ID: {}".format(markerID))

                # show the output image
                # cv2.imshow("Image", image)
            # print("TopLeft: ("+str(topLeftX)+","+str(topLeftY)+")")
            # print("TopRight: ("+str(topRightX)+","+str(topRightY)+")")
            # print("BottomLeft: ("+str(bottomLeftX)+","+str(bottomLeftY)+")")
            # print("BottomRight: ("+str(bottomRightX)+","+str(bottomRightY)+")")
            croppedimage = image[topLeftY:bottomRightY,topLeftX:bottomRightX]

            #imagebw = cv2.imread('stickersdots/my2.jpg', cv2.IMREAD_GRAYSCALE)
            #edges = cv.Canny(croppedimage,0.4,5)
            cv2.imshow("Image",croppedimage)
            print(randint(0,10000))
            #datatoadd = get_data(image,croppedimage,newpath,randint(0,10000))
            image = PIL.Image.fromarray(croppedimage)
            annotations = ocrmac.OCR(image).recognize()
            annotations_ordered = sort_by_Y(annotations)

            #id, original image
            #att: make and model, annotations
            #croppedimage,newpath 
            datatoadd = first_approach_read(croppedimage,newpath,images,
            exif_data['Make']+" "+exif_data['Model'],annotations_ordered,
            randint(0,1000000))
            write_to_csv("metadata_alfie_7aug.csv",datatoadd)

           # write_to_csv("metadata_representation.csv",datatoadd)
            

            #cv2.waitKey(0)
        #break