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
        print("-----my:"+str(annotations[i]))
        print("-----my:"+str(annotations[i][2][2]))
        if annotations[i][2][2]>0.12 and annotations[i][2][2]<0.3:
            print( annotations[i][0])
            return annotations[i][0]
        

def get_data(imagetodisplay,imagecropped,newfolder,originalimage,archive_code,c):
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
    alldataarray.insert(count,newpath+"/"+fullnamefile)
    count +=  1 
    
    alldataarray.insert(count,originalimage)
    count +=  1    
    
    alldataarray.insert(count,archive_code)
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
    if (images.endswith(".jpg")):
        print(images)
        # images to add
        #imagetodisplay = folder_dir+"/"+"IMG_20240724_135949~2.jpg"#images
        #display_image(imagetodisplay)
        image = cv2.imread(folder_dir+"/"+images)
        image = imutils.resize(image, width=800)

        #image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

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


                #print("centre: ("+str(cX)+","+str(cY)+")")
                #cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                ## draw the ArUco marker ID on the image
                #cv2.putText(image, str(markerID),
                #	(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                #	0.5, (0, 255, 0), 2)
                #print("[INFO] ArUco marker ID: {}".format(markerID))

                #show the output image
                #cv2.imshow("Image", image)
            # print("TopLeft: ("+str(topLeftX)+","+str(topLeftY)+")")
            # print("TopRight: ("+str(topRightX)+","+str(topRightY)+")")
            # print("BottomLeft: ("+str(bottomLeftX)+","+str(bottomLeftY)+")")
            # print("BottomRight: ("+str(bottomRightX)+","+str(bottomRightY)+")")
            croppedimage = image[topLeftY:bottomRightY,topLeftX:bottomRightX]

            #imagebw = cv2.imread('stickersdots/my2.jpg', cv2.IMREAD_GRAYSCALE)
            #edges = cv.Canny(croppedimage,0.4,5)
            #cv2.imshow("Image",image)
            #print(randint(0,10000))

            datatoadd = get_data(image,croppedimage,newpath,images,archive_code,randint(0,1000000))
            write_to_csv("metadata_representation_7aug.csv",datatoadd)
            #cv2.waitKey(0)
        #break