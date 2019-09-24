# facialRecognitionTerminalProgram
Command Line Facial Recognition Program

v1.3

This program uses the following libraries: openCV, numpy, dlib, face_recognition

face_recognition is a facial recognition wrapper for dlib which can be found at:
https://github.com/ageitgey/face_recognition

This program uses a Haar Cascade for face detection

To use this program first run: "python3 setup.py" to install all
required libraries

Commands for this program are:

1. python3 main.py addface image_path : adds a face encoding to the database
2. python3 main.py run : runs webcam face recognition
3. python3 main.py help : prints this menu
4. python3 main.py scanphoto image_path : scans a photo for face recognition
5. python3 main.py addfacehog image_path : adds a face encoding to the database
 using HOG face detection
6. python3 main.py runhog : runs webcam face recognition using HOG face
 detection
7. python3 main.py scanphotohog image_path : scans a photo for face recognition
 using HOG face detection

When adding an image or scanning one
image_path must be a path to a .jpg file with exactly
one face in the image or else the facial recognition will not work properly

TO DO:
- Bug Check
- Allow multiple photos to be uploaded at once
