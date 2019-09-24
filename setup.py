import os

#pip3 install a library
def install(package):
    os.system("pip3 install " + package)

#Installs required libraries
def main():
    packages = ["numpy", "opencv-python", "dlib", "face_recognition"]

    for package in packages:
        install(package)

main()
