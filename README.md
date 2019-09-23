# facialRecognitionTerminalProgram
Command Line Facial Recognition Program

v1.0

This program uses the following librares: openCV, numpy, dlib, face_recognition
face_recognition is a facial recognition wrapper for dlib which can be found at:
https://github.com/ageitgey/face_recognition

To use this program first run: "python3 setup.py" to install all
required libraries

Commands for this program are:

1. python3 main.py addface image_path : adds a face encoding to the database
2. python3 main.py run : runs webcam face recognition
3. python3 main.py help : prints this menu

When adding an image image_path must be a path to a .jpg file with exactly
one face in the image or else the facial recognition will not work properly

TO DO:
- Bug Check
- Allow Photo Recognition not just Webcam
- Allow multiple photos to be uploaded at once
