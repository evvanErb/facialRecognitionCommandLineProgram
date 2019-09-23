#Python 3.x

import face_recognition
import numpy as np

MAX_DISTANCE = 0.6

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
