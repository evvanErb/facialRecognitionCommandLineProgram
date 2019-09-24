#Python 3.x

import face_recognition
import numpy as np
import cv2

from facialDetection import haarDetectFaceLocations, hogDetectFaceLocations

MAX_DISTANCE = 0.6

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

def getFaceEncodings(image, faces):
    """
    Run the embedding model to get face embeddings for the supplied locations
    """
    faceEncodings = face_recognition.face_encodings(image, faces)

    return faceEncodings

def recognizeFace(faceEncoding, knownFaceEncodings, knownFaceNames):
    """
    Compare face encoding to all known face encodings and find the
    closest match and return their name
    """
    #Get the distances from this encoding to
    #those of all reference images
    distances = face_recognition.face_distance(knownFaceEncodings,
        faceEncoding)

    #Select the closest match (smallest distance)
    #if it's below the threshold value
    if np.any(distances <= MAX_DISTANCE):
        bestMatchIdx = np.argmin(distances)
        name = knownFaceNames[bestMatchIdx]
    else:
        name = None

    return name

def detectAndRecognizeFacesInImage(image,
    knownFaceEncodings, knownFaceNames, useHOG=False, isBGR=False):
    """
    Detects and recognizies faces in image then pains recognition info on image
    """
    #Detect if there are any faces in the frame and get their locations
    if (useHOG):
        faceLocations = hogDetectFaceLocations(image)
    else:
        faceLocations = haarDetectFaceLocations(image)

    #Get detected faces encoding from embedding model
    faceEncodings = getFaceEncodings(image, faceLocations)

    #Loop through each face in the frame and see if there's a match
    for location, faceEncoding in zip(faceLocations, faceEncodings):

        name = recognizeFace(faceEncoding, knownFaceEncodings,
            knownFaceNames)

        #Put recognition info on the image
        paintDetectedFaceOnImage(image, location, name, isBGR)
