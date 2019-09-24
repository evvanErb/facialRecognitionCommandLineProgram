#Python 3.x

import face_recognition
import cv2
import numpy as np
import glob
import os
import sys

from facialDetection import haarDetectFaceLocations, hogDetectFaceLocations
from facialRecognition import getFaceEncodings, recognizeFace

DATABASE_PATH = 'facialDatabase/'
CAMERA_DEVICE_ID = 0

def addPhoto(fileName, useHOG=False):
    """
    Load a supplied photo and add detected facial encoding to the database
    """
    #Check if image is a jpg
    if (fileName[-4:] != ".jpg"):
        print("\n[!] File extenstion must be .jpg!\n")
        return

    elif (not os.path.isfile(fileName)):
        print("\n[!] File does not exist!\n")
        return

    #Load image
    image = face_recognition.load_image_file(fileName)

    #Use the name in the filename as the identity key
    identity = os.path.splitext(os.path.basename(fileName))[0]

    #Get the face encoding
    if (useHOG):
        locations = hogDetectFaceLocations(image)
    else:
        locations = haarDetectFaceLocations(image)

    encodings = getFaceEncodings(image, locations)

    #Save data to file
    np.savetxt((DATABASE_PATH + identity + ".txt"), encodings[0])

def setupDatabase():
    """
    Load reference images and create a database of their face encodings
    """
    database = {}

    for filename in glob.glob(os.path.join(DATABASE_PATH, '*.txt')):
        #Use the name in the filename as the identity key
        identity = os.path.splitext(os.path.basename(filename))[0]

        #Get the face encoding and link it to the identity
        encoding = np.loadtxt(filename)

        database[identity] = encoding

    return list(database.values()), list(database.keys())

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

def runScanPhotoFaceRecognition(fileName, useHOG=False):
    """
    Manages facial recogntion on photos
    """
    #Check if image is a jpg
    if (fileName[-4:] != ".jpg"):
        print("\n[!] File extenstion must be .jpg!\n")
        return

    elif (not os.path.isfile(fileName)):
        print("\n[!] File does not exist!\n")
        return

    #Setup database
    knownFaceEncodings, knownFaceNames = setupDatabase()

    #Load image
    image = face_recognition.load_image_file(fileName)

    #Run facial detection and recognition on image
    detectAndRecognizeFacesInImage(image,
        knownFaceEncodings, knownFaceNames, useHOG, True)

    #Convert image from BGR to RGB and display the resulting image
    image = image[:, :, ::-1]
    cv2.imshow(fileName, image)

    print("\n[*] Press Q to quit\n")

    #Hit 'q' on the keyboard to quit!
    cv2.waitKey(0)


def runFaceRecognition(useHOG=False):
    """
    Manages live facial recognition
    """
    #Open a handler for the camera
    video_capture = cv2.VideoCapture(CAMERA_DEVICE_ID)

    #Setup database
    knownFaceEncodings, knownFaceNames = setupDatabase()

    skipFrame = 0

    while video_capture.isOpened():
        #Skip every other frame to increase frame rate
        if (skipFrame == 1):
            skipFrame = 0
            continue
        else:
            skipFrame = 1

        #Read frame from camera and check that it went ok
        ok, frame = video_capture.read()
        if not ok:
            print("[!] Error reading frame from camera. Video capture stopped.")
            break

        #Run facial detection and recognition on image
        detectAndRecognizeFacesInImage(frame,
            knownFaceEncodings, knownFaceNames, useHOG)

        #Display the resulting image
        cv2.imshow('Video', frame)

        #Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


def main():
    #Check if there is an argument
    if (len(sys.argv) < 2):
        print("\n[!] No arguments!\n")
        return

    #Get user argument
    argument = sys.argv[1]

    if (argument == "addface"):
        #If user didnt supply a photo path
        if (len(sys.argv) < 3):
            print("\n[!] No photo path!\n")
            return

        #Otherwise add photo to database
        photoPath = sys.argv[2]
        addPhoto(photoPath)

    elif (argument == "addfacehog"):
        #If user didnt supply a photo path
        if (len(sys.argv) < 3):
            print("\n[!] No photo path!\n")
            return

        #Otherwise add photo to database
        photoPath = sys.argv[2]
        addPhoto(photoPath, True)

    elif (argument == "run"):
        print("\n[*] Press Q to quit\n")
        runFaceRecognition()

    elif (argument == "runhog"):
        print("\n[*] Press Q to quit\n")
        runFaceRecognition(True)

    elif (argument == "scanphoto"):
        #If user didnt supply a photo path
        if (len(sys.argv) < 3):
            print("\n[!] No photo path!\n")
            return

        #Otherwise add photo to database
        photoPath = sys.argv[2]
        runScanPhotoFaceRecognition(photoPath)

    elif (argument == "scanphotohog"):
        #If user didnt supply a photo path
        if (len(sys.argv) < 3):
            print("\n[!] No photo path!\n")
            return

        #Otherwise add photo to database
        photoPath = sys.argv[2]
        runScanPhotoFaceRecognition(photoPath, True)

    elif (argument == "help"):
        print("\nArguments for Live Facial Recognition Software include:\n")
        print("1. python3 main.py addface image_path : adds ", end="")
        print("a face encoding to the database")
        print("2. python3 main.py run : runs webcam face recognition")
        print("3. python3 main.py help : prints this menu")
        print("4. python3 main.py scanphoto image_path : ", end="")
        print("scans a photo for face recognition")
        print("5. python3 main.py addfacehog image_path : adds ", end="")
        print("a face encoding to the database using HOG face detection")
        print("6. python3 main.py runhog : runs webcam face ", end="")
        print("recognition using HOG face detection")
        print("7. python3 main.py scanphotohog image_path : ", end="")
        print("scans a photo for face recognition using HOG face detection\n")

    else:
        print("\n[!] Unknown argument!\n")


main()
