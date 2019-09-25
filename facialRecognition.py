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

def recognizeFace(faceEncoding, knownFaceEncodings):
    """
    Compare face encoding to all known face encodings for this person and
    find the close matches and return their count
    """
    #Get the distances from this encoding to
    #those of all reference images for this person
    distances = face_recognition.face_distance(knownFaceEncodings,
        faceEncoding)

    possibleMathcesCount = 0

    #Look at all matches that have a distance below the MAX_DISTANCE
    #if it's below the threshold value then add +1 to this persons match count
    for distance in distances:
        if (distance <= MAX_DISTANCE):
            possibleMathcesCount += 1

    return possibleMathcesCount

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

        matches = {}

        #Iterate over all people in the database face encodings and get
        #how many photos per known person matched the unknown face
        for person in database:

            personMatchCount = recognizeFace(faceEncoding, database[person])

            matches[person] = personMatchCount

        #Iterate over all matches and see who has highest count
        bestMatch = None
        bestMatchCount = 0
        for match in matches:
            if ((matches[match] > 0) and (matches[match] > bestMatchCount)):
                bestMatch = match
                bestMatchCount = matches[match]

        #Put recognition info on the image
        paintDetectedFaceOnImage(image, location, bestMatch, isBGR)
