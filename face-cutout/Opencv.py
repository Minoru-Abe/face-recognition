import os
import subprocess
from PIL import Image
import cv2 as cv

dir2 = './org'
dir3 = './face'

files2 = os.listdir(dir2)
files2.sort()

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

for file in files2:
    if '.jpg' in file or '.JPG' in file:
        dirfile = os.path.join(dir2, file)
        img = cv.imread(dirfile)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        print(file)

        for (x,y,w,h) in faces:
            face = img[y-10:y+h+10, x-10:x+w+10]
            face_name = str(file.strip('.jpg'))+'_'+str(x)+'_'+str(y)+'.jpg'
            dirface = os.path.join(dir3,face_name)
            facefile = cv.imwrite(dirface, face)
            #cv.rectangle(img,(x-10,y-10),(x+w+10,y+h+10),(255,0,0),2)
            print(face_name)
