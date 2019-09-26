#Python 3.x

import face_recognition
import cv2
import numpy as np
import os

from settings import *
from facialDetection import haarDetectFaceLocations, hogDetectFaceLocations
from facialRecognition import numberOfMatches, recognizeFace

def detectAndRecognizeFacesInImage(image,
    database, useHOG=False, isBGR=False):
    """
    Detects and recognizies faces in image then paints recognition info on image
    """
    #Detect if there are any faces in the frame and get their locations
    if (useHOG):
        faceLocations = hogDetectFaceLocations(image)
    else:
        faceLocations = haarDetectFaceLocations(image)

    #Get detected faces encoding from embedding model
    faceEncodings = face_recognition.face_encodings(image, faceLocations)

    #Loop through each face in the frame and see if there's a match
    for location, faceEncoding in zip(faceLocations, faceEncodings):

        #See who from database matches best
        bestMatch = recognizeFace(database, faceEncoding)

        #Put recognition info on the image
        paintDetectedFaceOnImage(image, location, bestMatch, isBGR)

def setupDatabase():
    """
    Load reference images and create a database of their face encodings
    """
    database = {}

    #iterate over subdirs in DATABASE_PATH
    for subdir, dirs, files in os.walk(DATABASE_PATH):
        #Use the name of the sub dir as the identity key
        identity = subdir[len(DATABASE_PATH) : ]
        #initialize this persons data set array to store all their encodings
        encodings = []

        #iterate over files in the specific persons data set
        for file in files:
            #only use if its the desired file type
            try:
                #Get the face encoding and link it to the identity
                encoding = np.loadtxt(subdir + "/" + file)
                #add this encoding to the persons array of encodings
                encodings.append(encoding)
            except:
                print("\n[*] Warning: Unreadable File in Database.\n")

        #add the person's encodings to the database
        database[identity] = encodings

    return database

def paintDetectedFaceOnImage(frame, location, name=None, isBGR=False):
    """
    Paint a rectangle around the face and write the name
    """
    #Unpack the coordinates from the location tuple
    top, right, bottom, left = location

    if name is None:
        name = 'Unknown'
        color = (0, 0, 255)  #Red for unrecognized face
        if (isBGR):
            color = (255, 0, 0)  #Red for unrecognized face
    else:
        color = (0, 128, 0)  #Dark green for recognized face

    #Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    #Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom),
        color, cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom - 6),
        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
