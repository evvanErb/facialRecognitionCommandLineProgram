#Python 3.x

import cv2

def convertToFrLocationFormat(faceLocations):
    """
    converts face locations into format for face_recognition
    """
    toReturn = []

    for (x, y, w, h) in faceLocations:
        top = y
        right = x + w
        bottom = y + h
        left = x
        toReturn.append((top, right, bottom, left))

    return toReturn

def detectFaceLocations(image):
    """
    Take a raw image and run both the face detection and face embedding model on it
    """

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Our operations on the frame come here
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image with openCV Haar Cascade
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    #convert face locations into face_recognition format
    faces = convertToFrLocationFormat(faces)

    return faces
