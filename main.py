#Python 3.x

import face_recognition
import cv2
import numpy as np
import glob
import os
import sys

from facialDetection import haarDetectFaceLocations, hogDetectFaceLocations
from facialRecognition import detectAndRecognizeFacesInImage

DATABASE_PATH = 'facialDatabase/'
CAMERA_DEVICE_ID = 0

def addPhoto(fileName, personName, useHOG=False):
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

    #Get the face location
    if (useHOG):
        locations = hogDetectFaceLocations(image)
    else:
        locations = haarDetectFaceLocations(image)

    #Get the face encoding
    encodings = face_recognition.face_encodings(image, locations)

    #check if exactly one face is in the photo
    if (len(encodings) == 0):
        print("[!] No face detected in the provided photo")
        return

    elif (len(encodings) > 1):
        print("[!] More than one face detected in the provided photo")
        return

    #Set path to respective dataset
    directoryToAddTo = DATABASE_PATH + personName

    #Look for directory
    exists = False
    for subdir, dirs, files in os.walk(DATABASE_PATH):
        if (subdir == directoryToAddTo):
            exists = True

    #If directory doesnt exist make it
    if (not exists):
        os.mkdir(directoryToAddTo)

    #Save data to file
    np.savetxt((directoryToAddTo + "/" + identity + ".txt"), encodings[0])

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
            #Get the face encoding and link it to the identity
            encoding = np.loadtxt(subdir + "/" + file)
            #add this encoding to the persons array of encodings
            encodings.append(encoding)

        #add the person's encodings to the database
        database[identity] = encodings

    return database

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
    database = setupDatabase()

    #Load image
    image = face_recognition.load_image_file(fileName)

    #Run facial detection and recognition on image
    detectAndRecognizeFacesInImage(image,
        database, useHOG, True)

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
    database = setupDatabase()

    skipFrame = 0

    while video_capture.isOpened():
        #Skip every 2 frames to increase frame rate
        if (skipFrame < 2):
            skipFrame += 1
            continue
        else:
            skipFrame = 0

        #Read frame from camera and check that it went ok
        ok, frame = video_capture.read()
        if not ok:
            print("[!] Error reading frame from camera. Video capture stopped.")
            break

        #Run facial detection and recognition on image
        detectAndRecognizeFacesInImage(frame,
            database, useHOG)

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
        if (len(sys.argv) < 4):
            print("\n[!] Not enough arguments!\n")
            return

        #Otherwise add photo to database
        photoPath = sys.argv[2]
        name = sys.argv[3]
        addPhoto(photoPath, name)

    elif (argument == "addfacehog"):
        #If user didnt supply a photo path
        if (len(sys.argv) < 4):
            print("\n[!] Not enough arguments!\n")
            return

        #Otherwise add photo to database
        photoPath = sys.argv[2]
        name = sys.argv[3]
        addPhoto(photoPath, name, True)

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
        print("1. python3 main.py addface image_path person_name:", end="")
        print(" adds a face encoding to the database")
        print("2. python3 main.py run : runs webcam face recognition")
        print("3. python3 main.py help : prints this menu")
        print("4. python3 main.py scanphoto image_path : ", end="")
        print("scans a photo for face recognition")
        print("5. python3 main.py addfacehog image_path person_name:", end="")
        print(" adds a face encoding to the database using HOG face detection")
        print("6. python3 main.py runhog : runs webcam face ", end="")
        print("recognition using HOG face detection")
        print("7. python3 main.py scanphotohog image_path : ", end="")
        print("scans a photo for face recognition using HOG face detection\n")

    else:
        print("\n[!] Unknown argument!\n")


main()
