#Python 3.x

import cv2
import face_recognition

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

def haarDetectFaceLocations(image):
    """
    Take a raw image and run the haar cascade face detection on it
    """

    #Create the haar cascade
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    #Our operations on the frame come here
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Detect faces in the image with openCV Haar Cascade
    faceLocations = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    #Convert face locations into face_recognition format
    faceLocations = convertToFrLocationFormat(faceLocations)

    return faceLocations

def hogDetectFaceLocations(image, isBGR=False):
    """
    Take a raw image and run the hog face detection on it
    """

    #Convert from BGR to RGB if needed
    if (isBGR):
        image = image[:, :, ::-1]

	#Run the face detection model to find face locations
    faceLocations = face_recognition.face_locations(image)

    return faceLocations
