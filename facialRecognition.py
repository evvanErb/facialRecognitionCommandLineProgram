#Python 3.x

import face_recognition
import numpy as np
import cv2

from settings import *

def numberOfMatches(faceEncoding, knownFaceEncodings):
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

def recognizeFace(database, faceEncoding):
    matches = {}

    #Iterate over all people in the database face encodings and get
    #how many photos per known person matched the unknown face
    for person in database:

        personMatchCount = numberOfMatches(faceEncoding, database[person])

        matches[person] = personMatchCount

    #Iterate over all matches and see who has highest count
    bestMatch = None
    bestMatchCount = 0
    for match in matches:
        if ((matches[match] > 0) and (matches[match] > bestMatchCount)):
            bestMatch = match
            bestMatchCount = matches[match]

    return bestMatch
